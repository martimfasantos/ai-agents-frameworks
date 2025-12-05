from __future__ import annotations
import asyncio
from dataclasses import dataclass

from rich.prompt import Prompt
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from dotenv import load_dotenv

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic Graph with the following features:
- State object persisting throughout entire workflow execution
- Nodes reading and modifying shared state via ctx.state
- Graph-based control flow with typed node transitions
- Human-in-the-loop to interact with the workflow

Graphs and finite state machines (FSMs) are a powerful abstraction to model, 
execute, control and visualize complex workflows.

While this library is developed as part of Pydantic AI; it has no dependency 
on `pydantic-ai` and can be considered as a pure graph-based state machine 
library.

NOTE: This example doesn't use GenAI capabilities, but we can create Nodes that
leverage LLMs as part of their internal logic. 
For an example, see https://ai.pydantic.dev/graph/#genai-example

Reference: https://ai.pydantic.dev/graph/#stateful-graphs
-----------------------------------------------------------------------
"""

# --- 1. Define state object that persists throughout the workflow ---
@dataclass
class MachineState:  
    """State object shared across all nodes in the graph."""
    user_balance: float = 0.0
    product: str | None = None

# --- 2. Define workflow nodes ---
@dataclass
class InsertCoin(BaseNode[MachineState]):  
    """Initial node: Prompts user to insert coins."""
    async def run(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted:  
        return CoinsInserted(float(Prompt.ask("Insert coins")))  

@dataclass
class CoinsInserted(BaseNode[MachineState]):
    """Node: Updates balance and decides next step based on state."""
    amount: float  

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> SelectProduct | Purchase:  
        # Modify shared state (persists across node executions)
        ctx.state.user_balance += self.amount
        print(f"💵 Inserted ${self.amount:.2f}")
        print(f"   Balance: ${ctx.state.user_balance:.2f}")
        
        # Conditional routing based on state
        if ctx.state.product is not None:  
            return Purchase(ctx.state.product)
        else:
            return SelectProduct()

# --- 3. Define product catalog ---
PRODUCTS = {  
    "water": 1.25,
    "soda": 1.50,
    "crisps": 1.75,
    "chocolate": 2.00,
}

# --- 4. Define remaining workflow nodes ---
@dataclass
class SelectProduct(BaseNode[MachineState]):
    """Node: Prompts user to select a product."""
    async def run(self, ctx: GraphRunContext[MachineState]) -> Purchase:
        print("Available products:")
        for product, price in PRODUCTS.items():
            print(f" - {product}: ${price:.2f}")
        return Purchase(Prompt.ask("Select product"))

@dataclass
class Purchase(BaseNode[MachineState, None, None]):  
    """Node: Attempts purchase and routes to appropriate next node."""
    product: str

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> End | InsertCoin | SelectProduct:
        if price := PRODUCTS.get(self.product):  
            # Store selected product in state
            ctx.state.product = self.product  
            
            if ctx.state.user_balance >= price:  
                # Complete purchase
                ctx.state.user_balance -= price
                print(f"✅ Purchased {self.product}!")
                print(f"💰 Change returned: ${ctx.state.user_balance:.2f}")
                # Return End node to terminate the graph
                return End("Enjoy your purchase!")
            else:
                # Insufficient funds - loop back to InsertCoin
                diff = price - ctx.state.user_balance
                print(f"⚠️  Insufficient funds for {self.product}")
                print(f"   Need ${diff:0.2f} more")
                return InsertCoin()  
        else:
            # Invalid product - loop back to SelectProduct
            print(f"❌ No such product: {self.product}, try again")
            return SelectProduct()  


# --- 5. Create the graph with all node types ---
vending_machine_graph = Graph(  
    nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)


# --- 6. Run the stateful graph workflow ---
async def main():
    
    print("=== Stateful Graph Example ===\n")
    print("Vending Machine Workflow")
    print("=" * 60)
    
    # Initialize state object (shared across all nodes)
    state = MachineState()
    
    # Run the workflow starting from InsertCoin node
    # The graph will execute nodes sequentially, following the control flow
    # defined by each node's return type until an End node is reached
    result = await vending_machine_graph.run(InsertCoin(), state=state)
    
    print("\n" + "=" * 60)
    print(f"🎉 Result: {result.output}")
    print(f"📊 Final state:")
    print(f"   Balance: ${state.user_balance:.2f}")
    print(f"   Product: {state.product}")
    
    print("Mermaid Diagram of Graph:")
    print(vending_machine_graph.mermaid_code(start_node=InsertCoin))


if __name__ == '__main__':
    asyncio.run(main())
