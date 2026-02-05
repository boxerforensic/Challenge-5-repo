import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


# ---------------------------
# Session state init
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "_last_ai_answer" not in st.session_state:
    st.session_state["_last_ai_answer"] = ""

# 파일 바뀌면 리셋하기 위한 키
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None

# "I'm ready!"를 매번 안 찍히게 제어
if "_ready_shown" not in st.session_state:
    st.session_state["_ready_shown"] = False


# ---------------------------
# Memory (✅ 항상 getter로만 접근해서 KeyError 방지)
# ---------------------------
def get_memory() -> ConversationBufferMemory:
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    return st.session_state["memory"]


def load_chat_history(_):
    memory = get_memory()
    return memory.load_memory_variables({})["chat_history"]


# ---------------------------
# UI helpers
# ---------------------------
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for m in st.session_state["messages"]:
        send_message(m["message"], m["role"], save=False)


# ---------------------------
# Streaming Callback
# ---------------------------
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        if self.message_box is not None:
            self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, role="ai")
        st.session_state["_last_ai_answer"] = self.message


# ---------------------------
# Retriever / Embedding
# ---------------------------
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_bytes: bytes, file_name: str, api_key_hash: str):
    file_path = f"./.cache/files/{file_name}"
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    raise RuntimeError("embed_file() should be called via embed_file_with_key()")


def embed_file_with_key(file_bytes: bytes, file_name: str, api_key: str):
    import hashlib

    api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]

    @st.cache_data(show_spinner="Embedding file...", hash_funcs={bytes: lambda b: hashlib.sha256(b).hexdigest()})
    def _embed(file_bytes_inner: bytes, file_name_inner: str, api_key_hash_inner: str):
        file_path = f"./.cache/files/{file_name_inner}"
        with open(file_path, "wb") as f:
            f.write(file_bytes_inner)

        cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name_inner}")

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )

        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()

    return _embed(file_bytes, file_name, api_key_hash)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n"
            "Use BOTH the uploaded file context and the chat history to answer.\n"
            "If the user asks about the conversation itself (e.g., 'what was my first question?'), "
            "answer using the chat history.\n"
            "If you don't know, say you don't know. Don't make anything up.\n\n"
            "Context:\n{context}\n\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


st.set_page_config(page_title="Challenge-5", page_icon="📃")
st.title("DocumentGPT")
st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])
    api_key = st.text_input("Come on Input Your AI Key", type="password")
    reset = st.button("Reset chat")


if reset:
    st.session_state["messages"] = []
    st.session_state["_last_ai_answer"] = ""
    st.session_state["_ready_shown"] = False
    st.session_state["active_file"] = None
    st.session_state.pop("memory", None)
    st.rerun()


if file and api_key:
    # 파일이 바뀌면 리셋
    if st.session_state.get("active_file") != file.name:
        st.session_state["active_file"] = file.name
        st.session_state["messages"] = []
        st.session_state["_last_ai_answer"] = ""
        st.session_state["_ready_shown"] = False
        st.session_state.pop("memory", None)
        _ = get_memory()  # 새 메모리 생성

    file_bytes = file.getvalue()
    retriever = embed_file_with_key(file_bytes, file.name, api_key)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        openai_api_key=api_key,
        callbacks=[ChatCallbackHandler()],
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "chat_history": RunnableLambda(load_chat_history),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    if not st.session_state["_ready_shown"]:
        send_message("I'm ready! Ask away!", "ai", save=False)
        st.session_state["_ready_shown"] = True

    paint_history()

    user_msg = st.chat_input("Ask anything about your file...")
    if user_msg:
        send_message(user_msg, "human")

        memory = get_memory()
        memory.chat_memory.add_user_message(user_msg)

        with st.chat_message("ai"):
            _ = chain.invoke(user_msg)

        ai_answer = st.session_state.get("_last_ai_answer", "")
        if ai_answer:
            memory.chat_memory.add_ai_message(ai_answer)

else:
    pass