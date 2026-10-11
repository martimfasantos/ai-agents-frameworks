import os
from typing import Any, Callable, NotRequired

from deepagents import RubricMiddleware, create_deep_agent
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------
In this example, we explore Deep Agents with the following features:
- prepare_messages_for_grader: reshape the transcript the grader sees
- build_grader_state + grader_state_schema: pass custom fields to the grader
- grader_middleware: read those fields inside the nested grader agent

RubricMiddleware runs a nested grader agent over the transcript. By
default that grader sees the whole conversation and nothing else. The
SDK integration hooks let you control both halves of its input: trim or
redact the messages it reads, and hand it extra state your own grading
policy depends on. Here the transcript is reduced to the final answer
and an internal note is redacted, while a house style guide is injected
into the grader's state and pulled into its prompt by middleware.

For more details, visit:
https://docs.langchain.com/oss/python/deepagents/middleware
-----------------------------------------------------------------------
"""


# --- 1. Reshape what the grader reads ---
def prepare_messages_for_grader(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Show the grader only the final answer, with internal notes redacted."""
    finals = [m for m in messages if isinstance(m, AIMessage) and m.text]
    kept = finals[-1:] if finals else []
    print(f"  [hook] grader sees {len(kept)} of {len(messages)} messages")
    redacted = []
    for message in kept:
        # Drop only the internal line — the lines after it still matter to the grader.
        lines = message.text.splitlines()
        kept_lines = [
            "[internal note redacted]" if line.strip().startswith("INTERNAL:") else line
            for line in lines
        ]
        if kept_lines != lines:
            print("  [hook] redacted an internal note")
        redacted.append(AIMessage(content="\n".join(kept_lines)))
    return redacted


# --- 2. A grader state schema with room for our own field ---
class StyleGraderState(AgentState):
    style_guide: NotRequired[str]


STYLE_GUIDE = (
    "House style: an answer satisfies the rubric only if its last line is exactly "
    "the token 'EOM'. Judge against this rule alone."
)


def build_grader_state(state: Any, iteration: int) -> dict[str, Any]:
    """Hand the grader the style guide it should grade against."""
    print(f"  [hook] injecting style guide for grader iteration {iteration}")
    return {"style_guide": STYLE_GUIDE}


# --- 3. Middleware that pulls the injected field into the grader's prompt ---
class StyleGuideMiddleware(AgentMiddleware):
    state_schema = StyleGraderState

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        guide = request.state.get("style_guide")
        if guide:
            request = request.override(
                system_prompt=f"{request.system_prompt}\n\n{guide}"
            )
        return handler(request)


# --- 4. Wire the hooks into RubricMiddleware ---
def on_evaluation(evaluation: dict[str, Any]) -> None:
    print(
        f"  [grader] iteration {evaluation['iteration']}: result={evaluation['result']}"
    )


rubric_middleware = RubricMiddleware(
    model=f"openai:{settings.OPENAI_MODEL_NAME}",
    prepare_messages_for_grader=prepare_messages_for_grader,
    build_grader_state=build_grader_state,
    grader_state_schema=StyleGraderState,
    grader_middleware=[StyleGuideMiddleware()],
    on_evaluation=on_evaluation,
    max_iterations=3,
)

agent = create_deep_agent(
    model=f"openai:{settings.OPENAI_MODEL_NAME}",
    system_prompt=(
        "Answer in one short sentence. Then add a line 'INTERNAL: drafted by the "
        "assistant', and finish with a final line containing only 'EOM'."
    ),
    middleware=[rubric_middleware],
)

# --- 5. Run with a rubric the style guide decides ---
print("=== Rubric grader integration hooks ===\n")

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "In one sentence, what is a deploy freeze?"}
        ],
        "rubric": "The answer must follow the house style guide provided to you.",
    }
)

# --- 6. Show the answer the grader accepted ---
final = result["messages"][-1]
print("\nFinal answer:")
print(f"  {final.text.strip()}")
print("\nThe grader never saw the INTERNAL note, and graded against a style")
print("guide that was never part of the conversation.")
