"""
Benchmark: Tool-Calling Reliability & Performance Comparison

Runs all 15 basic agents with a non-obvious prompt that requires tool calling
(date_tool). Measures response time, token usage, and tool-call success rate.

Usage:
    python benchmark_tool_calling.py --provider openai --iterations 10
"""

import importlib
import time
import json
import os
import argparse
import numpy as np
from datetime import datetime, date

# Ensure OPENAI_API_KEY env var is set for frameworks that require it
# (CrewAI, PydanticAI, OpenAI Agents SDK, LlamaIndex, etc.)
from settings import settings

if not os.environ.get("OPENAI_API_KEY") and settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()


# ---- All 15 Basic Agents ----
AGENTS = {
    "agno_agent": ("Agent", "Agno"),
    "ag2_agent": ("Agent", "AG2"),
    "claude_sdk_agent": ("Agent", "Claude SDK"),
    "crewai_agent": ("Agent", "CrewAI"),
    "google_adk_agent": ("Agent", "Google ADK"),
    "langchain_agent": ("Agent", "LangChain"),
    "langgraph_agent": ("Agent", "LangGraph"),
    "llama_index_agent": ("Agent", "LlamaIndex"),
    "llama_index_fc_agent": ("Agent", "LlamaIndex FC"),
    "microsoft_agent": ("Agent", "Microsoft AF"),
    "openai_agent": ("Agent", "OpenAI (Raw)"),
    "openai_agents_sdk_agent": ("Agent", "OpenAI Agents SDK"),
    "pydantic_ai_agent": ("Agent", "PydanticAI"),
    "smolagents_agent": ("Agent", "Smolagents"),
    "strands_agent": ("Agent", "Strands"),
}

# Non-obvious prompt that requires the date_tool
# The agent must infer it needs to call date_tool, then compute the day of the week
PROMPT = "What day of the week is it today?"


def detect_tool_call(response: str) -> bool:
    """
    Detect whether the date_tool was called by checking if the response
    contains today's actual date or the correct day of the week.

    date_tool returns: date.today().strftime("%B %d, %Y") -> e.g., "April 06, 2026"
    If the agent used the tool, it will know the actual date and can compute the day.
    """
    today = date.today()
    response_lower = response.lower()

    # Check for today's date in various formats
    date_indicators = [
        today.strftime("%B %d, %Y").lower(),  # "April 06, 2026"
        today.strftime("%B %-d, %Y").lower(),  # "April 6, 2026"
        today.strftime("%Y-%m-%d"),  # "2026-04-06"
        today.strftime("%d/%m/%Y"),  # "06/04/2026"
        today.strftime("%m/%d/%Y"),  # "04/06/2026"
        today.strftime("%A").lower(),  # "monday"
        today.strftime("%B %-d").lower(),  # "april 6"
        today.strftime("%B %d").lower(),  # "april 06"
    ]

    for indicator in date_indicators:
        if indicator in response_lower:
            return True

    return False


def load_agent(module_name: str, class_name: str, provider: str):
    """Dynamically load and instantiate an agent."""
    try:
        module = importlib.import_module(module_name)
        AgentClass = getattr(module, class_name)
        agent = AgentClass(
            provider=provider,
            memory=False,
            verbose=False,
            tokens=True,
        )
        return agent
    except Exception as e:
        print(f"  [ERROR] Failed to load {module_name}: {e}")
        return None


def run_benchmark(provider: str, iterations: int, filter_agents: list = None):
    """Run the benchmark across all agents."""
    results = {}

    for module_name, (class_name, display_name) in AGENTS.items():
        if filter_agents and module_name not in filter_agents:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {display_name} ({module_name})")
        print(f"{'=' * 60}")

        response_times = []
        token_counts = {
            "prompt_llm_token_count": [],
            "completion_llm_token_count": [],
            "total_llm_token_count": [],
        }
        tool_calls = 0
        errors = 0
        sample_response = ""

        for i in range(iterations):
            # Create a fresh agent for each iteration (no memory contamination)
            agent = load_agent(module_name, class_name, provider)
            if agent is None:
                errors += 1
                continue

            try:
                result = agent.chat(PROMPT)

                if isinstance(result, tuple) and len(result) == 3:
                    response, exec_time, tokens = result
                elif isinstance(result, tuple) and len(result) == 2:
                    response, exec_time = result
                    tokens = {}
                else:
                    response = str(result)
                    exec_time = 0.0
                    tokens = {}

                response_str = str(response)

                # Track response time
                response_times.append(exec_time)

                # Track tokens
                if tokens:
                    for key in token_counts:
                        token_counts[key].append(tokens.get(key, 0))

                # Detect if tool was called
                if detect_tool_call(response_str):
                    tool_calls += 1

                # Store first response as sample
                if i == 0:
                    sample_response = response_str[:300]

                status = "TOOL" if detect_tool_call(response_str) else "NO TOOL"
                print(
                    f"  Run {i + 1}/{iterations}: {exec_time:.2f}s [{status}] {response_str[:80]}..."
                )

            except Exception as e:
                errors += 1
                print(f"  Run {i + 1}/{iterations}: ERROR - {str(e)[:100]}")

        # Aggregate results
        agent_result = {
            "display_name": display_name,
            "module_name": module_name,
            "iterations": iterations,
            "errors": errors,
            "tool_calls": tool_calls,
            "tool_call_rate": f"{tool_calls}/{iterations}",
            "sample_response": sample_response,
        }

        if response_times:
            agent_result["response_time_mean"] = float(np.mean(response_times))
            agent_result["response_time_std"] = float(np.std(response_times))
        else:
            agent_result["response_time_mean"] = 0.0
            agent_result["response_time_std"] = 0.0

        if any(token_counts["total_llm_token_count"]):
            agent_result["avg_prompt_tokens"] = float(
                np.mean(token_counts["prompt_llm_token_count"])
            )
            agent_result["avg_completion_tokens"] = float(
                np.mean(token_counts["completion_llm_token_count"])
            )
            agent_result["avg_total_tokens"] = float(
                np.mean(token_counts["total_llm_token_count"])
            )
        else:
            agent_result["avg_prompt_tokens"] = 0.0
            agent_result["avg_completion_tokens"] = 0.0
            agent_result["avg_total_tokens"] = 0.0

        results[display_name] = agent_result

        # Print agent summary
        print(
            f"\n  Summary: {tool_calls}/{iterations} tool calls | "
            f"{agent_result['response_time_mean']:.2f} +/- {agent_result['response_time_std']:.2f}s | "
            f"tokens: {agent_result['avg_total_tokens']:.1f} | "
            f"errors: {errors}"
        )

    return results


def print_results_table(results: dict):
    """Print a markdown-formatted results table."""
    print(f"\n\n{'=' * 100}")
    print(f'  BENCHMARK RESULTS — Prompt: "{PROMPT}"')
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"{'=' * 100}\n")

    # Sort by display_name
    sorted_results = sorted(results.values(), key=lambda x: x["display_name"])

    # Print markdown table
    print(
        "| Framework | Response Time | Avg Tokens (prompt/completion/total) | Tool Calls | Errors |"
    )
    print("|---|---|---|---|---|")

    for r in sorted_results:
        time_str = f"{r['response_time_mean']:.2f} +/- {r['response_time_std']:.2f}s"
        tokens_str = f"{r['avg_prompt_tokens']:.0f} / {r['avg_completion_tokens']:.0f} / {r['avg_total_tokens']:.0f}"
        print(
            f"| {r['display_name']} | {time_str} | {tokens_str} | {r['tool_call_rate']} | {r['errors']} |"
        )

    print()


def save_results(results: dict, output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"tool_calling_benchmark_{timestamp}.json")

    output = {
        "prompt": PROMPT,
        "timestamp": datetime.now().isoformat(),
        "today": date.today().isoformat(),
        "results": results,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Tool-Calling Benchmark")
    parser.add_argument(
        "--provider", type=str, choices=["azure", "openai", "other"], default="openai"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--agents", nargs="*", help="Specific agent module names to test"
    )
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"  Tool-Calling Benchmark")
    print(f"  Provider: {args.provider}")
    print(f"  Iterations: {args.iterations}")
    print(f'  Prompt: "{PROMPT}"')
    print(f"  Today: {date.today().strftime('%A, %B %d, %Y')}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    results = run_benchmark(args.provider, args.iterations, args.agents)
    print_results_table(results)
    save_results(results, args.output)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
