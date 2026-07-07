import os
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
  

#STEP 1: Bulding RAG
def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1200 , chunk_overlap = 260)

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embeddings) 

    return vectorstore.as_retriever(search_type="mmr",search_kwargs = {"k":8,"fetch_k":20})

acedemic_retriver = build_retriver("academics_handbook.pdf")
fee_retriver = build_retriver("fee_structure.pdf")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.4,
)

#Step 2 -  State

class State(TypedDict):
    programme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str


#Step 3 - Nodes generation

def classifire_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take"""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic' , 'fee' ,'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits,"
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    catagory = response.content.strip().lower()

    if "academic" in catagory:
        catagory = "academic"
    elif "fee" in catagory:
        catagory = "fee"
    else:
        catagory = "general"

    return {"query_type" : catagory}


#step 4 creating a node

def acadmic_rag_node(state:State)->dict:
    "Retrives relevent chunks from the acadmics handbook."
    query = state["messages"][-1].content
    docs = acedemic_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state:State)->dict:
    "Retrives relevent chunks from the fee structure PDF."
    query = state["messages"][-1].content
    programme = state["programme"]      # <-- ADD THIS
    search_query = f"{programme} {query}"   # <-- ADD THIS
    docs = fee_retriver.invoke(search_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state:State) -> dict:
    """Answer dircticly using the LLM's own language, no retrival needed."""
    return {"retrieved_context":"No_Retrival_needed"}


def respons_node(state:State) -> dict:
    """Generat the final answer, personalized using the student's progeamme."""
    query = state["messages"][-1].content
    programme = state.get( "programme","Unknown" )
    context = state["retrieved_context"]

    if context == "No_Retrival_needed":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student."
            f"Answer this question using your own general knowledge: \n\n{query}"

        )
    else:
        prompt= (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Give a clear, friendly, and precise answer."
            "IMPORTANT:\n"
            "Answer ONLY from the context.\n"
            "If the answer is not present in the context, say "
            "'I couldn't find this information in the provided documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"

        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


#step 5 router function

def router_query(state:State) :
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state["query_type"] == 'fee':
        return "fee_rag"
    else:
        return "general"
    
#step 6 Building The Graph

graph = StateGraph(State)

graph.add_node("classifire",classifire_node)
graph.add_node("academic_rag",acadmic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",respons_node)

#edges

graph.add_edge(START,"classifire")


graph.add_conditional_edges(
    "classifire" , router_query
)

graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)

app = graph.compile()
#Run the code
if __name__ == "__main__":

    print("Welcome to the Collage assistant")
    print("which program are you in ? ")
    print("1. BCA")
    print("2. BBA")
    print("3. B.com (H)")

    choice = input("\n Enter 1,2 or 3")

    programme_map = {
        "1" : "BCA",
        "2" : "BBA",
        "3" : "B.Com (H)"
    }
    student_program = programme_map.get(choice,"BCA")

    print(f"\nGreat! You're set as a {student_program} student!")

    while True:
        user_qyery = input("You:  ")

        if user_qyery.lower() in ["exit","quit"]:
            break

        result = app.invoke({
            "programme" : student_program,
            "messages":[("human",user_qyery)]
        })

        print(f"Assistant : {result['messages'][-1].content}")

