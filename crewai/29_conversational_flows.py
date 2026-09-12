import os

from crewai.flow import Flow, listen
from crewai.flow.conversational import ConversationConfig, ConversationState, RouterConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- Conversational Flows, promoted from experimental to stable in 1.15.18
- conversational = True plus ConversationConfig to opt a Flow into chat
- Custom routes as @listen("label") handlers, picked by the LLM router
- The built-in converse and end routes, and ConversationState history

A regular Flow runs once from start to finish. A conversational Flow
instead runs one turn at a time: each user message is classified by an
LLM router into a route label, the matching handler runs, and its reply
is appended to a canonical message history that survives across turns.
Built-in routes cover ordinary chat and goodbyes, so you only write
handlers for the intents your domain actually has.

For more details, visit:
https://docs.crewai.com/en/concepts/flows
-------------------------------------------------------
"""


# --- 1. Mock order data a custom route can answer from ---
ORDERS = {
    "A-1001": "shipped, arriving Tuesday",
    "A-1002": "still being packed",
}


# --- 2. A conversational Flow ---
class SupportFlow(Flow[ConversationState]):
    # Opting in registers the built-in conversational graph (converse, end)
    # and turns handle_turn() into the per-turn entry point.
    conversational = True

    conversational_config = ConversationConfig(
        system_prompt=(
            "You are a terse order-support assistant. Answer in one short sentence."
        ),
        llm=settings.OPENAI_MODEL_NAME,
        router=RouterConfig(
            llm=settings.OPENAI_MODEL_NAME,
            # Descriptions are what the router sees when choosing a route.
            route_descriptions={
                "check_order": "The user asks about the status of a specific order id.",
            },
        ),
    )

    # --- 3. A custom route — declaring one auto-enables the LLM router ---
    @listen("check_order")
    def handle_check_order(self) -> str:
        """Look up an order id mentioned by the user."""
        message = (self.state.current_user_message or "").upper()
        for order_id, status in ORDERS.items():
            if order_id in message:
                return f"Order {order_id} is {status}."
        return "I need an order id like A-1001 to look that up."


# --- 4. Drive the conversation one turn at a time ---
# handle_turn() is the programmatic entry point; chat() is its REPL wrapper
# and is not used here so the example stays non-interactive.
print("=== Conversational Flow ===\n")

flow = SupportFlow()

turns = [
    "Hi, who am I talking to?",
    "What's the status of order A-1001?",
    "And A-1002?",
    "Great, that's all — bye!",
]

for turn in turns:
    reply = flow.handle_turn(turn)
    print(f"You:       {turn}")
    print(f"Route:     {flow.state.last_intent}")
    print(f"Assistant: {reply}\n")

# --- 5. History and end-of-conversation flag persist on the state ---
print(f"Messages recorded: {len(flow.state.messages)}")
print(f"Conversation ended: {flow.state.ended}")
