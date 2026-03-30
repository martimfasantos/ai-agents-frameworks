import os
from pydantic import BaseModel

from crewai.flow.flow import Flow, start, listen
from crewai.flow import human_feedback, HumanFeedbackProvider

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI Flows with the following features:
- @human_feedback decorator for human-in-the-loop workflows
- Custom HumanFeedbackProvider for non-interactive feedback
- Routing based on human approval/rejection
- Integrating human decisions into flow state

The @human_feedback decorator enables human-in-the-loop patterns
in Flows. A custom provider can supply feedback programmatically,
which is useful for testing, CI/CD, or automated approval systems.

For more details, visit:
https://docs.crewai.com/en/learn/human-feedback-in-flows
-------------------------------------------------------
"""


# --- 1. Create a custom feedback provider (no input() calls) ---
class AutoApprovalProvider:
    """A feedback provider that automatically approves proposals.

    Implements the HumanFeedbackProvider protocol. In production, this
    could call an external API, read from a queue, or integrate with a
    UI for real human review.
    """

    def request_feedback(self, context, flow) -> str:
        """Provide automated feedback based on context."""
        output = context.method_output
        print(f"[AutoApproval] Reviewing: {str(output)[:100]}...")
        # Simulate approval — in production, this would be a real review
        return "approved"


# --- 2. Define the flow state ---
class ProposalState(BaseModel):
    proposal: str = ""
    feedback: str = ""
    final_status: str = ""


# --- 3. Create a flow with human feedback ---
class ProposalFlow(Flow[ProposalState]):
    @start()
    def generate_proposal(self):
        """Generate a proposal for review."""
        self.state.proposal = (
            "Proposal: Implement a new caching layer to reduce API latency by 40%. "
            "Estimated effort: 2 weeks. Cost: $5,000."
        )
        print(f"Generated proposal: {self.state.proposal}")
        return self.state.proposal

    @human_feedback(
        message="Please review the proposal and respond with 'approved' or 'rejected':",
        provider=AutoApprovalProvider(),
        emit=["approved", "rejected"],
        default_outcome="approved",
    )
    @listen(generate_proposal)
    def review_proposal(self):
        """Present the proposal for human review."""
        return self.state.proposal

    @listen("approved")
    def handle_approval(self):
        """Handle approved proposals."""
        self.state.final_status = "APPROVED"
        print("Proposal was APPROVED! Proceeding with implementation.")
        return self.state.final_status

    @listen("rejected")
    def handle_rejection(self):
        """Handle rejected proposals."""
        self.state.final_status = "REJECTED"
        print("Proposal was REJECTED. Revisions needed.")
        return self.state.final_status


# --- 4. Run the flow ---
if __name__ == "__main__":
    flow = ProposalFlow()
    result = flow.kickoff()
    print(f"\nFinal status: {flow.state.final_status}")
    print(f"Flow result: {result}")
