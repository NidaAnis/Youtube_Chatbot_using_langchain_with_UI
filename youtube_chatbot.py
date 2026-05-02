import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="▶",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: #0a0a0f;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #ff4d6d, #ff9a3c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #7a7a9a;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

/* Input card */
.input-card {
    background: #13131f;
    border: 1px solid #2a2a3f;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Chat bubbles */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
}
.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #ff4d6d22, #ff9a3c22);
    border: 1px solid #ff4d6d55;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    max-width: 75%;
    font-size: 0.95rem;
}
.bubble-bot {
    align-self: flex-start;
    background: #13131f;
    border: 1px solid #2a2a3f;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
    opacity: 0.5;
}
.label-user { color: #ff9a3c; text-align: right; }
.label-bot  { color: #7a9fff; }

/* Status badge */
.status-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.badge-ready   { background: #0d2b1a; color: #3ddc84; border: 1px solid #3ddc8455; }
.badge-pending { background: #1a1a0d; color: #ffcc00; border: 1px solid #ffcc0055; }

/* Stray Streamlit elements */
.stTextInput > div > div > input {
    background: #1c1c2e !important;
    border: 1px solid #2a2a3f !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff4d6d !important;
    box-shadow: 0 0 0 2px #ff4d6d22 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ff4d6d, #ff9a3c) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid #1e1e30 !important;
}
.stSpinner > div { border-top-color: #ff4d6d !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return url_or_id.strip()


@st.cache_resource(show_spinner=False)
def build_chain(video_id: str, groq_api_key: str):
    # 1. Transcript
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id, languages=['en'])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])

    # 3. Embed + store
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    # 4. Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant. Answer only from the transcript context.
If the context is insufficient, just say you don't know.
{context}
Question: {question}
""",
        input_variables=['context', 'question']
    )

    # 5. LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    # 6. Chain
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        | prompt | llm | StrOutputParser()
    )
    return chain, len(chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. Paste a YouTube URL  
2. Click **Load Video**  
3. Ask anything about it  
    """)
    st.markdown("---")
    st.markdown(
        "<span style='font-family:Space Mono,monospace;font-size:0.72rem;color:#4a4a6a'>"
        "Embeddings: MiniLM-L6-v2<br>LLM: Llama 3.1 8B via Groq</span>",
        unsafe_allow_html=True
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>▶ YouTube Chatbot</h1>
  <p>RAG-powered · ask anything about any video</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Video loader ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    video_url = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=LPZh9BOjkQs",
        label_visibility="collapsed"
    )
with col2:
    load_btn = st.button("Load Video", use_container_width=True)

if load_btn:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not video_url:
        st.error("Please enter a YouTube URL or video ID.")
    else:
        vid_id = extract_video_id(video_url)
        with st.spinner(f"Fetching transcript & building index for `{vid_id}`…"):
            try:
                chain, n_chunks = build_chain(vid_id, groq_key)
                st.session_state.chain = chain
                st.session_state.history = []
                st.session_state.video_loaded = True
                st.session_state.chunk_count = n_chunks
                st.success(f"Ready! Indexed **{n_chunks}** chunks from the transcript.")
            except Exception as e:
                st.error(f"Error: {e}")

# Status badge
if st.session_state.video_loaded:
    st.markdown(
        f'<span class="status-badge badge-ready">● READY · {st.session_state.chunk_count} chunks indexed</span>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="status-badge badge-pending">○ NO VIDEO LOADED</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Chat area ─────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"""
            <div class="bubble-user">
              <div class="label label-user">YOU</div>
              {msg}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bubble-bot">
              <div class="label label-bot">BOT</div>
              {msg}
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Question input ────────────────────────────────────────────────────────────
if st.session_state.video_loaded:
    q_col, btn_col = st.columns([5, 1])
    with q_col:
        user_q = st.text_input(
            "Ask a question",
            placeholder="Summarize the video / What is discussed about transformers?",
            label_visibility="collapsed",
            key="question_input"
        )
    with btn_col:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and user_q:
        st.session_state.history.append(("user", user_q))
        with st.spinner("Thinking…"):
            try:
                answer = st.session_state.chain.invoke(user_q)
                st.session_state.history.append(("bot", answer))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.markdown(
        "<p style='color:#4a4a6a;font-family:Space Mono,monospace;font-size:0.82rem;"
        "text-align:center;padding:1.5rem'>Load a video above to start chatting.</p>",
        unsafe_allow_html=True
    )

# Clear chat
if st.session_state.history:
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.rerun()
