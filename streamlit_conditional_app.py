import streamlit as st
import conditional_RAG

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #eef1f5 100%);
    }

    /* Main title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4f46e5 0%, #7c3aed 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.15);
        border-radius: 10px;
        color: #ffffff;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Chat input box */
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1px solid #d1d5db !important;
    }

    /* Footer badge in sidebar */
    .sidebar-footer {
        position: fixed;
        bottom: 20px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="main-title">🎓 College AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your personal guide for programme queries, admissions, and more</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown("### 📚 Select Your Programme")
program = st.sidebar.selectbox(
    "Select Programme",
    [
        "BCA",
        "BBA",
        "B.Com (H)"
    ],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 **Tip:** Ask about eligibility, fees, subjects, career paths, or admission dates."
)
st.sidebar.markdown('<div class="sidebar-footer">Powered by Conditional RAG ⚡</div>', unsafe_allow_html=True)

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            result = conditional_RAG.app.invoke(
                {
                    "programme": program,
                    "messages": [("human", prompt)]
                }
            )
            answer = result["messages"][-1].content
            st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )