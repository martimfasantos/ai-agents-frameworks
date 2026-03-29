import os
import random
from pydantic import BaseModel

from crewai.flow.flow import Flow, listen, start, router, and_, or_
from crewai.flow.persistence import persist

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Structured state management using Pydantic models
- @start() decorator for flow entry points
- @listen() decorator for method chaining and event listening
- @router() decorator for conditional routing logic
- and_() function for waiting on multiple conditions
- or_() function for listening to multiple events
- State persistence and management throughout the flow
- Flow visualization with automatic plot generation (in "res/" folder)
- Multiple execution paths with conditional branching

When using Flows, CrewAI provides powerful AI workflow orchestration with:
1. Event-driven architecture for responsive workflows
2. Flexible control flow with conditional logic and routing
3. Robust state management for data sharing between tasks
4. Automatic visualization for understanding flow structure

For better visualization, I recommend running the code, check the terminal logs,
and then view the generated flow plot in the "res/" folder.

For more details, visit:
https://docs.crewai.com/en/concepts/flows
-------------------------------------------------------
"""


# --- 1. Define a structured state ---
class FlowState(BaseModel):
    counter: int = 0
    message: str = ""
    decision: str = ""
    final_result: str = ""


# --- 2. Create a Flow with various features ---
@persist(verbose=True)  # 2.1 Class-level persistence with logging
class SimpleFlowExample(Flow[FlowState]):
    # 2.2 Start method using @start()
    # @persist()  # Optional method-level persistence
    @start()
    def initialize_flow(self):
        """Start method that initializes the flow"""
        print("🚀 Starting simple flow example")
        print(f"📋 Flow State ID: {self.state.id}")
        self.state.message = "Initial message"
        self.state.counter = 1
        return "initialization_complete"

    # 2.3 Listener method using @listen()
    @listen(initialize_flow)
    def update_message(self, initialization_result):
        """Listen to initialization and update state"""
        print(f"📝 Received: {initialization_result}")
        self.state.message += " - updated by second method"
        self.state.counter += 1
        return "message_updated"

    @listen(update_message)
    def make_decision(self):
        """Make a random decision for routing"""
        print("🤔 Making a decision...")
        choices = ["path_a", "path_b"]
        self.state.decision = random.choice(choices)
        print(f"🎯 Decision made: {self.state.decision}")
        return self.state.decision

    # 2.4 Router method using @router() for conditional routing
    @router(make_decision)
    def route_based_on_decision(self):
        """Router that returns different paths based on decision"""
        if self.state.decision == "path_a":
            return "path_a"
        elif self.state.decision == "path_b":
            return "path_b"

    # 2.5 Listener methods for each path
    @listen("path_a")
    def handle_path_a(self):
        """Handle path A"""
        print("🛣️  Taking Path A")
        self.state.message += " - Path A taken"
        self.state.counter += 5
        return "path_a_completed"

    @listen("path_b")
    def handle_path_b(self):
        """Handle path B"""
        print("🛣️  Taking Path B")
        self.state.message += " - Path B taken"
        self.state.counter += 10
        return "path_b_completed"

    # 2.6 AND condition - waits for multiple paths
    @listen(and_(handle_path_a, handle_path_b))
    def wait_for_all_paths(self):
        """Wait for all possible paths (though only one will execute)"""
        print("⏳ This would wait for all paths, but only one executes due to routing")
        return "all_paths_considered"

    # 2.7 OR condition - listens to multiple methods
    @listen(or_(handle_path_a, handle_path_b))
    def handle_any_path_completion(self, path_result):
        """Handle completion of any path"""
        print(f"✅ Path completed: {path_result}")
        self.state.final_result = (
            f"Completed {path_result} with counter: {self.state.counter}"
        )
        return self.state.final_result

    # 2.8 Final listener method
    @listen(handle_any_path_completion)
    def finalize_flow(self, final_result):
        """Final method that processes the result"""
        print("🎉 Flow completed!")
        print(f"📊 Final state: {self.state}")
        print(f"🏁 Final result: {final_result}")
        return f"SUCCESS: {final_result}"


# --- 3. Run the flow ---
if __name__ == "__main__":
    # 3.1 Create flow instance
    flow = SimpleFlowExample()

    # 3.2 Generate visualization
    flow.plot("res/simple_flow_plot")

    # 3.3 Execute the flow
    result = flow.kickoff()

    print("\n" + "=" * 50)
    print("FINAL OUTPUT:")
    print(f"Result: {result}")
    print(f"Final State: {flow.state}")
    print("=" * 50)
