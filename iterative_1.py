import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from groq import BadRequestError
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearch(max_results=3)

tools = [search_tool]

# llms

# writer
writer_llm = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.4,  # lowered from 0.7 - higher temps make Groq's tool-call
                       # formatting much more likely to break (see writer_node)
)
writer_llm_with_tools = writer_llm.bind_tools(tools)


# reviewer
reviewer_llm = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

# state building


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# Building Nodes

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context. "
    "Only use the Tavily search tool if the user explicitly asks for: "
    "latest news, current trends, today's information, or recent statistics. "
    "When you do call the search tool, pass only a simple 'query' string - "
    "do not add extra parameters. "
    "If you have already received feedback on a previous draft, carefully "
    "address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150-200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}. "
            f"It must be 150-200 words, open with a hook that creates "
            f"curiosity or tension (not a generic 'what is X' question), "
            f"and include at least one concrete example or specific detail. "
            f"If you need current info, search the web first."
        )
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected. "
            f"Here is the reviewer's feedback:\n\n{previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentioned. "
            f"If the feedback mentions word count, treat it as the top priority - "
            f"actually change the length accordingly, don't just resubmit a "
            f"similarly-sized draft. Do not repeat the same mistakes."
        )
    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]

    # Groq's llama-3.3-70b-versatile occasionally emits a malformed / raw-text
    # function call (instead of a proper structured tool call), which the
    # Groq API rejects with a 400 "Failed to call a function" error. This is
    # a known intermittent issue, not something in our control on every call,
    # so we retry a couple of times, then fall back to a no-tools generation
    # rather than letting the whole graph crash.
    response = None
    max_retries = 2
    for attempt_no in range(max_retries + 1):
        try:
            response = writer_llm_with_tools.invoke(messages)
            break
        except BadRequestError as e:
            print(f"[writer_node] Tool-call generation failed "
                  f"(attempt {attempt_no + 1}/{max_retries + 1}): {e}")

    if response is None:
        print("[writer_node] Falling back to generation without tool use.")
        response = writer_llm.invoke(messages)

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }


tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state["messages"][-1]
    draft = last_message.content
    print(f"\n\nGenerated post:\n{draft}\n")

    return {"draft": draft}


REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line - it must create curiosity, tension, "
    "or surprise. Generic openers like 'What is X?' or 'Let's talk about X' "
    "count as WEAK and should be rejected.\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim - uses short paragraphs\n"
    "4. Word count must be between 150 and 200 words. Count the words "
    "yourself and state the count in your response. Reject if outside "
    "this range.\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n"
    "8. Includes at least one concrete detail, example, or specific angle - "
    "not just generic statements\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "WORD_COUNT: <number>\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state["draft"]

    prompt = (
        f"Review this LinkedIn post draft:\n"
        f"{draft}\n"
        f"Give your review."
    )
    response = reviewer_llm.invoke(
        [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]
    )
    review_text = response.content.strip()

    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    # Don't rely solely on the LLM's self-reported word count - it's not
    # always accurate (it sometimes claims the count is "within range" even
    # when it isn't). Enforce the 150-200 word range in code, and if it
    # fails, replace the feedback entirely with a clear, actionable message
    # instead of appending to text that may directly contradict it.
    word_count = len(draft.split())
    word_count_ok = 150 <= word_count <= 200

    if not word_count_ok:
        is_approved = False
        if word_count < 150:
            shortfall = 150 - word_count
            feedback = (
                f"REJECTED - word count only: this draft is {word_count} words, "
                f"which is {shortfall} words short of the required 150-200 range. "
                f"Keep the same structure and message, but expand it by adding "
                f"one more concrete example, a brief supporting detail, or an "
                f"extra sentence developing the takeaway. Do not just reword the "
                f"same length draft - it must be noticeably longer than last time."
            )
        else:
            excess = word_count - 200
            feedback = (
                f"REJECTED - word count only: this draft is {word_count} words, "
                f"which is {excess} words over the 150-200 range. Trim redundant "
                f"or repetitive sentences without losing the hook, takeaway, or CTA."
            )

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Word count: {word_count}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }


# Router function


def should_use_tool(state: State):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        print("Post has been approved.\n")
        return END
    if state["attempt"] >= 4:
        print("Reached max attempts.")
        return END
    return "writer"


# build the graph
graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)


graph.add_edge(START, "writer")
graph.add_conditional_edges("writer", should_use_tool)
graph.add_edge("tools", "writer")
graph.add_edge("extract_draft", "reviewer")

graph.add_conditional_edges("reviewer", should_stop_looping)

app = graph.compile()

print("=" * 55)
print("Welcome to the LinkedIn post generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")
print("=" * 55)


topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}")
