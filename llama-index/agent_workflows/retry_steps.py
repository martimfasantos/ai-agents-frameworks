import asyncio
import random
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent
from workflows.retry_policy import ConstantDelayRetryPolicy, ExponentialBackoffRetryPolicy


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Automatic retry on step failure
- ConstantDelayRetryPolicy for fixed retry intervals
- ExponentialBackoffRetryPolicy for increasing delays
- Custom retry policies for specific failure scenarios

Retry policies help handle transient failures like network timeouts,
rate limits, or temporary service unavailability.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/retry_steps/
-------------------------------------------------------
"""


class UnreliableServiceError(Exception):
    """Simulates a transient service error"""
    pass


class RetryWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempt_count = 0

    @step(retry_policy=ConstantDelayRetryPolicy(delay=1, maximum_attempts=3))
    async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Step with retry policy - simulates transient failures"""
        self.attempt_count += 1
        
        # Simulate a service that fails randomly
        if random.random() < 0.7 and self.attempt_count < 3:
            raise UnreliableServiceError("Service temporarily unavailable")
        
        return StopEvent(result=f"Succeeded after {self.attempt_count} attempts")


async def main():
    workflow = RetryWorkflow(timeout=30, verbose=False)
    result = await workflow.run()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
