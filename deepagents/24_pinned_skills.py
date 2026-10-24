import re
import textwrap

from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Deep Agents with the following features:
- Pinning skills with the pinned_skills run input
- Detecting a /skill-name reference in the user's message
- Spotting the injected pinned-skill message via additional_kwargs

Normally the agent reads a skill only when it decides the skill matches.
When the user names a skill, for example by typing /haiku-writer, the
app can pin it instead: SkillsMiddleware appends the SKILL.md body to the
conversation before the next model call, so the instructions are in
context without the agent choosing them. The same message is sent twice:
unpinned, the model may or may not open the skill on its own; pinned, it
always gets the instructions, proven by the skill's "(fin)" marker.

For more details, visit:
https://docs.langchain.com/oss/python/deepagents/skills#pin-skills
-----------------------------------------------------------------------
"""

# --- 1. A skill whose instructions leave a visible fingerprint ---
SKILL_MD = """---
name: haiku-writer
description: Writes a strict 5-7-5 syllable haiku.
---
# Haiku Writer Skill
Reply with exactly three lines following 5-7-5 syllables.
End the haiku with the marker (fin).
"""
seed_files = {"/skills/haiku-writer/SKILL.md": create_file_data(SKILL_MD)}

# --- 2. A regular skills-enabled agent; pinning needs no extra setup ---
agent = create_deep_agent(
    model=f"openai:{settings.OPENAI_MODEL_NAME}",
    skills=["/skills/"],
)

# --- 3. Deep Agents never parses messages; the app finds /skill-name itself ---
SKILL_REFERENCE = re.compile(r"(?<!\S)/([a-z0-9-]+)")
message = "/haiku-writer the first frost of autumn"


def run(pin: bool) -> None:
    """Send the message, pinning any /skill-name it references when pin is True."""
    pinned_skills = SKILL_REFERENCE.findall(message) if pin else []
    print(f"[Run with pinned_skills={pinned_skills}]")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": message}],
            "files": seed_files,
            "pinned_skills": pinned_skills,
        }
    )

    # --- 4. The injected message is tagged so a UI can render it as a label ---
    for msg in result["messages"]:
        if msg.additional_kwargs.get("lc_source") == "pinned_skill":
            skill = msg.additional_kwargs["skill"]
            print(f"  [pinned skill injected] name={skill['name']} path={skill['path']}")
    reads = [
        tc["args"]["file_path"]
        for msg in result["messages"]
        if msg.type == "ai"
        for tc in msg.tool_calls
        if tc["name"] == "read_file"
    ]
    print(f"  read_file calls: {reads or 'none'}")

    reply = result["messages"][-1].text
    print(f"  Agent:\n{textwrap.indent(reply.strip(), '    ')}")
    print(f"  Followed the skill ('(fin)' marker present): {'(fin)' in reply}\n")


# --- 5. Send the same message without and with pinning ---
print("=== Deep Agents Pinned Skills ===")
print(f"User: {message}\n")
run(pin=False)
run(pin=True)
