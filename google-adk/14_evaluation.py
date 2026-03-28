import os
import sys
import json
import asyncio
import logging
import tempfile
import textwrap

from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

# Suppress the SDK's "non-text parts in response" informational warning
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- AgentEvaluator: programmatic evaluation of agent behavior
- Test cases: JSON-based eval datasets with expected tool trajectories
- Evaluation criteria: tool_trajectory_avg_score and response_match_score
- Automated scoring: comparing actual vs expected tool usage and responses

The ADK evaluation framework lets you define test cases (user queries,
expected tool calls, and expected responses) then automatically score
your agent against them. This is essential for regression testing —
ensuring that changes to agents or models don't silently break behavior.

For more details, visit:
https://google.github.io/adk-docs/evaluate/
-------------------------------------------------------
"""


# --- 1. Build the agent module in a temporary directory ---
#
# AgentEvaluator.evaluate() loads an agent via Python's import system.
# The module must be importable and expose a `root_agent` inside an `agent`
# sub-module. We create this structure on the fly in a temp directory.


def create_agent_module(tmp_dir: str, model_name: str) -> None:
    """Write an importable agent package to tmp_dir."""
    pkg_dir = os.path.join(tmp_dir, "dice_agent")
    os.makedirs(pkg_dir, exist_ok=True)

    # __init__.py — expose the `agent` sub-module
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write("from . import agent\n")

    # agent.py — define tools and root_agent.
    # Note: check_prime returns dict[str, bool] because the Genai SDK
    # requires function response dict keys to be strings.
    agent_code = textwrap.dedent(f"""\
        import random
        from google.adk.agents import LlmAgent

        def roll_die(sides: int) -> int:
            \"\"\"Roll a single die with the given number of sides.\"\"\"
            return random.randint(1, sides)

        def check_prime(nums: list[int]) -> dict[str, bool]:
            \"\"\"Check whether each number in nums is prime.\"\"\"
            def is_prime(n: int) -> bool:
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            return {{str(n): is_prime(n) for n in nums}}

        root_agent = LlmAgent(
            name="DiceAgent",
            model="{model_name}",
            instruction=(
                "You are a dice-rolling assistant. "
                "Use roll_die to roll dice and check_prime to check if numbers are prime. "
                "Always use tools when asked. Reply concisely in one sentence."
            ),
            tools=[roll_die, check_prime],
        )
    """)
    with open(os.path.join(pkg_dir, "agent.py"), "w") as f:
        f.write(agent_code)


# --- 2. Build evaluation test cases ---
#
# Each eval case specifies a user query and what the agent is expected to do:
# - `intermediate_data.tool_uses`: the expected tool call(s) (name + args)
# - `final_response`: a reference answer for ROUGE-1 similarity scoring
#
# For tool_trajectory_avg_score, the framework does exact matching on
# tool names and arguments. For response_match_score it uses ROUGE-1.


def create_eval_dataset(tmp_dir: str) -> str:
    """Write a .test.json eval dataset and return its path."""
    eval_data = {
        "eval_set_id": "dice_agent_eval_set",
        "name": "Dice Agent Evaluation",
        "description": "Tests that the dice agent calls the correct tools.",
        "eval_cases": [
            {
                "eval_id": "roll_6_sided_die",
                "conversation": [
                    {
                        "invocation_id": "inv-001",
                        "user_content": {
                            "parts": [{"text": "Roll a 6-sided die."}],
                            "role": "user",
                        },
                        # response_match_score uses ROUGE-1 (word overlap).
                        # Keep the reference short and similar to what the agent says.
                        "final_response": {
                            "parts": [{"text": "You rolled a number."}],
                            "role": "model",
                        },
                        "intermediate_data": {
                            "tool_uses": [{"name": "roll_die", "args": {"sides": 6}}],
                            "intermediate_responses": [],
                        },
                    }
                ],
                "session_input": {
                    "app_name": "dice_agent",
                    "user_id": "test_user",
                    "state": {},
                },
            },
            {
                "eval_id": "check_if_7_is_prime",
                "conversation": [
                    {
                        "invocation_id": "inv-002",
                        "user_content": {
                            "parts": [{"text": "Is 7 a prime number?"}],
                            "role": "user",
                        },
                        "final_response": {
                            "parts": [{"text": "Yes, 7 is a prime number."}],
                            "role": "model",
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {"name": "check_prime", "args": {"nums": [7]}}
                            ],
                            "intermediate_responses": [],
                        },
                    }
                ],
                "session_input": {
                    "app_name": "dice_agent",
                    "user_id": "test_user",
                    "state": {},
                },
            },
        ],
    }

    eval_path = os.path.join(tmp_dir, "dice_agent.test.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    return eval_path


# --- 3. Write an evaluation config ---
#
# test_config.json sits in the same directory as the .test.json file.
# It sets the scoring thresholds:
#   - tool_trajectory_avg_score: 1.0 means exact match on tool names/args
#   - response_match_score: 0.3 is lenient — ROUGE-1 on short varied responses
#     is inherently noisy since the die result is random each run.


def create_eval_config(tmp_dir: str) -> None:
    """Write test_config.json alongside the test file."""
    config = {
        "criteria": {
            "tool_trajectory_avg_score": 1.0,
            "response_match_score": 0.3,
        }
    }
    config_path = os.path.join(tmp_dir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# --- 4. Run the evaluation ---


async def main() -> None:
    from google.adk.evaluation.agent_evaluator import AgentEvaluator

    print("-" * 55)
    print("       Agent Evaluation Framework Demo")
    print("-" * 55)

    with tempfile.TemporaryDirectory() as tmp_dir:
        create_agent_module(tmp_dir, settings.GOOGLE_MODEL_NAME)
        eval_path = create_eval_dataset(tmp_dir)
        create_eval_config(tmp_dir)

        # Add temp dir to sys.path so Python can import the dice_agent package
        sys.path.insert(0, tmp_dir)
        try:
            print("\nEvaluating 2 test cases:")
            print("  Case 1: Roll a 6-sided die  (expected tool: roll_die)")
            print("  Case 2: Is 7 prime?         (expected tool: check_prime)")
            print()
            print("Criteria:")
            print("  tool_trajectory_avg_score >= 1.0  (exact tool-call match)")
            print("  response_match_score      >= 0.3  (ROUGE-1 word overlap)")
            print()

            try:
                await AgentEvaluator.evaluate(
                    agent_module="dice_agent",
                    eval_dataset_file_path_or_dir=eval_path,
                    num_runs=1,
                    print_detailed_results=True,
                )
                # AgentEvaluator only prints details on failures.
                # Reaching here means all cases passed.
                print()
                print(
                    "  [PASSED] tool_trajectory_avg_score — agent called the correct tools"
                )
                print(
                    "  [PASSED] response_match_score      — response matched reference"
                )
                print()
                print("All evaluation cases passed.")
            except AssertionError as e:
                # Some cases failed — the evaluator already printed the failure table.
                print()
                print(f"Some evaluation cases FAILED:\n{e}")
        finally:
            sys.path.remove(tmp_dir)


if __name__ == "__main__":
    asyncio.run(main())
