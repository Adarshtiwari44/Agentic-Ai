import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

print("API Key Loaded:", os.getenv("GROQ_API") is not None)

# =====================================================
# Initialize LLM
# =====================================================

llm = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

# =====================================================
# State Definition
# =====================================================

class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_txt: str
    final: str


# =====================================================
# Node 1 : Editor
# =====================================================

def editor_node(state: PipelineState):

    print("\n========== Editor Node ==========\n")

    prompt = f"""
You are a professional technical editor.

Your job is to improve the writing while preserving the author's original meaning.

Instructions:

- Correct grammar.
- Correct spelling.
- Improve sentence flow.
- Remove unnecessary repetition.
- Keep the tone natural.
- Do NOT exaggerate.
- Do NOT add new information.

Return ONLY the edited text.

Raw Text:

{state["raw_input"]}
"""

    response = llm.invoke(prompt)

    return {
        "edited_text": response.content.strip()
    }


# =====================================================
# Node 2 : Script Writer
# =====================================================

def scriptwriter_node(state: PipelineState):

    print("\n========== Script Writer Node ==========\n")

    prompt = f"""
You are an expert YouTube Script Writer.

Convert the following edited text into an engaging educational YouTube script.

Rules:

- Start with a powerful hook.
- Keep it conversational.
- Sound like an experienced educator.
- Build curiosity naturally.
- Explain clearly.
- Keep it concise.

Avoid:

- clickbait
- unnecessary hype
- "magic"
- "mind blowing"
- "superpower"
- "secret sauce"

Target length:
120-150 words.

Edited Text:

{state["edited_text"]}

Return ONLY the script.
"""

    response = llm.invoke(prompt)

    return {
        "script_txt": response.content.strip()
    }


# =====================================================
# Node 3 : Translator
# =====================================================

def translator_node(state: PipelineState):

    print("\n========== Translator Node ==========\n")

    prompt = f"""
You are an expert Indian content localizer.

Translate the following script into natural Hinglish.

Guidelines:

- Mix Hindi and English naturally.
- Keep technical words in English.
- Explain naturally in Hindi.
- Do NOT translate word-for-word.
- Keep the flow smooth.
- Sound like an Indian tech educator.

Avoid:

- overacting
- unnecessary excitement
- cringe language
- "magic"
- "superpower"

Return ONLY the Hinglish version.

Script:

{state["script_txt"]}
"""

    response = llm.invoke(prompt)

    return {
        "final": response.content.strip()
    }


# =====================================================
# Build LangGraph
# =====================================================

graph = StateGraph(PipelineState)

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

workflow = graph.compile()

# =====================================================
# Execute Workflow
# =====================================================

result = workflow.invoke(
    {
        "raw_input": """
AI agents are the future of technology.
They can think, plan, and act independently.
LangGraph helps developers build reliable AI agents
with memory, workflows, and complete control.
"""
    }
)

print("\n================ FINAL OUTPUT ================\n")
print(result["final"])