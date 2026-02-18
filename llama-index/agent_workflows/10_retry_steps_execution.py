import asyncio
import random
from typing import Optional
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent
from workflows.retry_policy import ConstantDelayRetryPolicy


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Automatic retry on step failure with retry policies
- ConstantDelayRetryPolicy for fixed-interval retries
- ExponentialBackoffRetryPolicy for increasing delays
- Writing custom retry policies using the RetryPolicy protocol

A step that fails might result in the entire workflow failing, but
transient errors (network timeouts, rate limits) can be safely retried.
Retry policies instruct the workflow to re-execute a step, controlling
the delay between attempts. Built-in policies include
ConstantDelayRetryPolicy (fixed delay, max attempts). 
The only requirement for custom policies is to write a Python class that 
respects the RetryPolicy protocol. In other words, implement a 
next(self, elapsed_time: float, attempts: int, error: Exception) -> Optional[float]
method: return a float (seconds to wait) or None (stop retrying).

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/retry_steps/
-------------------------------------------------------
"""


# --- 1. ConstantDelayRetryPolicy: fixed retry interval ---
class TransientError(Exception):
    """Simulates a transient service error"""
    pass


class ConstantRetryWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempt_count = 0

    # Retry every 1 second, up to 5 attempts
    @step(retry_policy=ConstantDelayRetryPolicy(delay=1, maximum_attempts=5))
    async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Step that fails randomly and retries with constant delay"""
        self.attempt_count += 1
        print(f"  Attempt {self.attempt_count}...")

        # Fail on the first 2 attempts
        if self.attempt_count < 3:
            raise TransientError(f"Transient failure on attempt {self.attempt_count}")

        return StopEvent(result=f"Succeeded after {self.attempt_count} attempts")


# --- 2. Custom retry policy using the RetryPolicy protocol ---
class RetryOnlySpecificErrorPolicy:
    """Custom policy that only retries on TransientError, gives up on others"""
    def next(
        self, elapsed_time: float, attempts: int, error: Exception
    ) -> Optional[float]:
        # Only retry TransientError, up to 3 times
        if isinstance(error, TransientError) and attempts < 3:
            return 0.5  # Retry in 0.5 seconds
        # Stop retrying for any other error or after max attempts
        return None


class CustomPolicyWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempt_count = 0

    @step(retry_policy=RetryOnlySpecificErrorPolicy())
    async def selective_retry_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Step with a custom retry policy that filters by error type"""
        self.attempt_count += 1
        print(f"  Attempt {self.attempt_count}...")

        if self.attempt_count == 1:
            raise TransientError("Transient issue — will retry")

        return StopEvent(result=f"Custom policy succeeded after {self.attempt_count} attempts")


# --- 3. Run all examples ---
async def main():
    print("=== Constant Delay Retry ===")
    w1 = ConstantRetryWorkflow(timeout=30, verbose=False)
    result = await w1.run()
    print(f"Result: {result}\n")

    print("=== Custom Retry Policy ===")
    w2 = CustomPolicyWorkflow(timeout=30, verbose=False)
    result = await w2.run()
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
