from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic Graph with the following features:
- State object persisting throughout entire workflow execution
- Nodes reading and modifying shared state via ctx.state
- Graph-based control flow with typed node transitions
- Conditional routing based on accumulated state

Graphs and finite state machines (FSMs) are a powerful abstraction to model,
execute, control and visualize complex workflows. While pydantic-graph is
developed as part of Pydantic AI, it has no dependency on pydantic-ai and
can be used as a standalone graph-based state machine library.

NOTE: This example does not use GenAI capabilities. See 15_graphs_with_genai.py
for an example that combines graph control flow with LLM-powered nodes.

For more details, visit:
https://ai.pydantic.dev/graph/
-----------------------------------------------------------------------
"""


# --- 1. Define state object that persists throughout the workflow ---
@dataclass
class MachineState:
    """State shared across all nodes in the vending machine graph."""

    user_balance: float = 0.0
    product: str | None = None
    coins_inserted: list[float] = field(default_factory=list)


# --- 2. Define the product catalog ---
PRODUCTS = {
    "water": 1.25,
    "soda": 1.50,
    "crisps": 1.75,
    "chocolate": 2.00,
}

# Simulated user actions (replaces interactive Prompt.ask)
SIMULATED_COINS = [1.00, 0.50, 0.50]
SIMULATED_PRODUCT = "soda"


# --- 3. Define workflow nodes ---
@dataclass
class InsertCoin(BaseNode[MachineState]):
    """Prompts user to insert coins (simulated)."""

    async def run(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted:
        # Take next coin from simulated input
        if ctx.state.coins_inserted:
            amount = ctx.state.coins_inserted.pop(0)
        else:
            amount = 0.25  # fallback
        return CoinsInserted(amount=amount)


@dataclass
class CoinsInserted(BaseNode[MachineState]):
    """Updates balance and decides next step based on state."""

    amount: float

    async def run(self, ctx: GraphRunContext[MachineState]) -> SelectProduct | Purchase:
        # Modify shared state (persists across node executions)
        ctx.state.user_balance += self.amount
        print(f"  Inserted ${self.amount:.2f}")
        print(f"   Balance: ${ctx.state.user_balance:.2f}")

        # Conditional routing based on state
        if ctx.state.product is not None:
            return Purchase(product=ctx.state.product)
        else:
            return SelectProduct()


@dataclass
class SelectProduct(BaseNode[MachineState]):
    """Selects a product (simulated)."""

    async def run(self, ctx: GraphRunContext[MachineState]) -> Purchase:
        print("Available products:")
        for product, price in PRODUCTS.items():
            print(f"   - {product}: ${price:.2f}")
        selected = SIMULATED_PRODUCT
        print(f"  Selected: {selected}")
        return Purchase(product=selected)


@dataclass
class Purchase(BaseNode[MachineState, None, None]):
    """Attempts purchase and routes to the appropriate next node."""

    product: str

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> End | InsertCoin | SelectProduct:
        price = PRODUCTS.get(self.product)
        if price is None:
            print(f"  No such product: {self.product}, try again")
            return SelectProduct()

        # Store selected product in state
        ctx.state.product = self.product

        if ctx.state.user_balance >= price:
            # Complete purchase
            ctx.state.user_balance -= price
            print(f"  Purchased {self.product}!")
            print(f"  Change returned: ${ctx.state.user_balance:.2f}")
            return End("Enjoy your purchase!")
        else:
            diff = price - ctx.state.user_balance
            print(f"  Insufficient funds for {self.product}")
            print(f"   Need ${diff:0.2f} more")
            return InsertCoin()


# --- 4. Create the graph with all node types ---
vending_machine_graph = Graph(
    nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)


# --- 5. Run the stateful graph workflow ---
async def main():
    print("=== Stateful Graph Example ===\n")
    print("Vending Machine Workflow")
    print("=" * 60)

    # Initialize state with simulated coin inserts
    state = MachineState(coins_inserted=list(SIMULATED_COINS))

    # Run the workflow starting from InsertCoin node
    # The graph executes nodes sequentially, following control flow
    # defined by each node's return type until an End node is reached
    result = await vending_machine_graph.run(InsertCoin(), state=state)

    print()
    print("=" * 60)
    print(f"Result: {result.output}")
    print(f"Final state:")
    print(f"   Balance: ${state.user_balance:.2f}")
    print(f"   Product: {state.product}")

    print("\nMermaid Diagram of Graph:")
    print(vending_machine_graph.mermaid_code(start_node=InsertCoin))


if __name__ == "__main__":
    asyncio.run(main())
