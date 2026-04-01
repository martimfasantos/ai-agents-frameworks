import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import query, ResultMessage

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- One-shot agent query using the query() function
- Iterating over message stream to get the final result

The simplest way to use the Claude Agent SDK: send a single prompt
and retrieve the result. The query() function returns an async iterator
of messages; we look for the ResultMessage to get the final answer.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/quickstart
-------------------------------------------------------
"""


# --- 1. Send a one-shot query ---
async def main():
    async for message in query(prompt="Where does 'hello world' come from?"):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


if __name__ == "__main__":
    asyncio.run(main())
