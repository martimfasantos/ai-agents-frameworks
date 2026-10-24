from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.eval import Case, run_cases
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.run import RunContext
from agno.scorer import CodeScorer, Score

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Eval Case run arguments: session_id, user_id, session_state
- Multi-turn evals: cases that share a session_id run as consecutive turns
- A Message (not just a str) as the case input

A Case without a session_id runs in a fresh, generated session, so it can only
grade a single-shot answer. Two cases with the same session_id become turns of
one conversation, so the second case grades whether the agent used what it
learned in the first — here, booking a call for a caller whose name was only
given one turn earlier.

For more details, visit:
https://docs.agno.com/evals/suite/overview
-------------------------------------------------------
"""

SESSION_ID = "booking-session-001"
USER_ID = "lead-42"
CALENDAR = {"calendar_id": "cal-7", "timezone": "Europe/Lisbon"}


# --- 1. Mock tools that read the run's session_state and user_id ---
def get_open_slots(run_context: RunContext) -> str:
    """Return the open appointment slots on the caller's calendar."""
    session_state = run_context.session_state
    if not session_state.get("calendar_id"):
        return "No calendar is configured for this caller."
    return f"Open slots ({session_state['timezone']}): Tuesday 10:00, Wednesday 15:00"


def create_appointment(run_context: RunContext, slot: str, name: str) -> str:
    """Book the given slot for the caller.

    Args:
        slot: The slot to book, as returned by get_open_slots.
        name: The caller's name.
    """
    run_context.session_state["appointment"] = {"slot": slot, "name": name}
    calendar_id = run_context.session_state.get("calendar_id")
    return f"Booked {slot} for {name} (user {run_context.user_id}) on calendar {calendar_id}."


# --- 2. Create the agent under test ---
# db + add_history_to_context let a turn read the earlier turns of its session.
agent = Agent(
    id="booking-agent",
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    db=InMemoryDb(),
    add_history_to_context=True,
    tools=[get_open_slots, create_appointment],
    instructions=[
        "You book sales calls. Use get_open_slots to offer times.",
        "Once the caller has given their name and picked a slot, call create_appointment.",
        "Reply in one short sentence.",
    ],
)


# --- 3. Deterministic scorers ---
def offers_both_slots(run, expected) -> bool:
    return "Tuesday" in run.content and "Wednesday" in run.content


def books_from_history(run, expected) -> Score:
    """Pass when create_appointment ran with the name from turn 1 and the picked slot."""
    calls = [tool.tool_args for tool in run.tools or [] if tool.tool_name == "create_appointment"]
    if not calls:
        return Score(value=0.0, passed=False, reason="create_appointment was not called")
    args = calls[0]
    passed = args.get("name") == expected["name"] and expected["day"] in args.get("slot", "")
    reason = f"booked name={args.get('name')!r}, slot={args.get('slot')!r}; expected {expected['name']!r} on {expected['day']}"
    return Score(value=1.0 if passed else 0.0, passed=passed, reason=reason)


def offers_no_slots(run, expected) -> bool:
    return "Tuesday" not in run.content and "Wednesday" not in run.content


# --- 4. Declare the cases ---
cases = [
    # Turn 1: the caller's calendar arrives through session_state.
    Case(
        name="turn 1: offers open slots",
        input="Hi, I'm Priya. I'd like to book a call. What times do you have?",
        agent=agent,
        expected_tool_calls=("get_open_slots",),
        scorer=CodeScorer(offers_both_slots),
        session_id=SESSION_ID,
        user_id=USER_ID,
        session_state=CALENDAR,
    ),
    # Turn 2: same session_id, so the agent sees turn 1. The name and the calendar
    # are not repeated — they can only come from the history and the stored session.
    Case(
        name="turn 2: books the slot using turn-1 context",
        input="Wednesday works for me, please book it.",
        agent=agent,
        scorer=CodeScorer(books_from_history),
        expected={"name": "Priya", "day": "Wednesday"},
        session_id=SESSION_ID,
        user_id=USER_ID,
    ),
    # Control: no session_id means a fresh session with no history and no calendar.
    # The input is a Message, which is how a case would attach images or files.
    Case(
        name="fresh session: no calendar, no slots",
        input=Message(role="user", content="What times do you have for a call?"),
        agent=agent,
        scorer=CodeScorer(offers_no_slots),
    ),
]

# --- 5. Run the suite ---
print("=== Running multi-turn eval suite ===\n")
suite_result = run_cases(cases)

# --- 6. Report each case ---
for result in suite_result.results:
    verdict = "PASS" if result.passed else "FAIL"
    print(f"[{verdict}] {result.name}")
    origin = "shared" if result.session_id == SESSION_ID else "generated per case"
    print(f"  session: {result.session_id} ({origin})")
    tools = result.response.tools if result.response else None
    for tool in tools or []:
        print(f"  tool:    {tool.tool_name}({tool.tool_args}) -> {tool.result}")
    print(f"  reply:   {result.output}")
    if result.score and result.score.reason:
        print(f"  score:   {result.score.reason}")
    if result.error:
        print(f"  error:   {result.error}")
    print()

# --- 7. Inspect the shared session the two turns ran in ---
session = agent.get_session(session_id=SESSION_ID)
print(f"=== Stored session '{SESSION_ID}' ===")
print(f"Runs in session: {len(session.runs or [])} (turn 1 + turn 2)")
print(f"Session state:   {agent.get_session_state(session_id=SESSION_ID)}")

print(f"\nSuite: {suite_result.status} — {suite_result.passed}/{suite_result.total} cases passed")
if suite_result.status != "PASS":
    raise SystemExit(1)
