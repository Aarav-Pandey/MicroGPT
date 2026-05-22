import datetime
import time
from pathlib import Path
import streamlit as st

from agent_controller import MicroGPTAgent
from rag_master import get_rag_engine

DOCS_PATH = Path("/home/aarav/MicroGPT/documents")
LOGO_FILE = Path(__file__).resolve().parent / "logo.svg"

st.set_page_config(
    page_title="MicroGPT | Local Agentic Inference",
    page_icon="🤖",
    layout="wide",
)

DOCS_PATH.mkdir(parents=True, exist_ok=True)

# SESSION INIT
if "agent" not in st.session_state:
    st.session_state.agent = MicroGPTAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {p.name for p in DOCS_PATH.iterdir() if p.is_file()}

# SIDEBAR
with st.sidebar:
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE))

    st.info("An Optimized Framework for Local Agentic Inference")
    st.divider()

    now = datetime.datetime.now()
    st.write(f"📅 {now.strftime('%A, %B %d, %Y')}")

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent.memory.clear_memory()
        st.rerun()

    st.divider()

    # FILE UPLOAD
    st.subheader("📤 Upload Documents")
    st.caption(f"Path: {DOCS_PATH}")

    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "txt", "md", "docx", "csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
        path = DOCS_PATH / uploaded_file.name
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_files.add(uploaded_file.name)
        st.success(f"{uploaded_file.name} uploaded successfully")
        # Auto-refresh RAG
        get_rag_engine().refresh_rag_index()
        st.rerun()

    st.divider()
    st.subheader("📂 Documents")
    files = sorted([p.name for p in DOCS_PATH.iterdir() if p.is_file()])

    if files:
        for file_name in files:
            col1, col2 = st.columns([4, 1])
            col1.write(file_name)
            if col2.button("❌", key=file_name):
                (DOCS_PATH / file_name).unlink()
                st.session_state.uploaded_files.discard(file_name)
                st.success(f"{file_name} deleted successfully")
                # Auto-refresh RAG
                get_rag_engine().refresh_rag_index()
                st.rerun()
    else:
        st.caption("No documents uploaded")

    st.divider()
    st.write(f"📊 Indexed Docs: {len(st.session_state.uploaded_files)}")

# MAIN UI
st.title("MicroGPT Agentic Interface")
st.caption("Local LLM | RAG | Math Tool")

# ✅ FIXED RAG STATUS (CLEAN START)
rag_engine = get_rag_engine()

if len(st.session_state.uploaded_files) > 0:
    if rag_engine.retriever is not None:
        st.success("RAG Active: Documents indexed and ready")
    else:
        pass

# CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.empty()

        full_response = ""
        start_time = time.time()

        for event in st.session_state.agent.run_stream(prompt):
            if event["type"] == "token":
                full_response += event["content"]
                placeholder.markdown(full_response)
            elif event["type"] == "status":
                status.info(event["message"])
        
        status.empty()
        st.caption(f"⚡ Latency: {round(time.time() - start_time, 2)}s")

    clean_response = full_response.strip()
    st.session_state.messages.append({"role": "assistant", "content": clean_response})