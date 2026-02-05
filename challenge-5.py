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


#session_state에 저장할 것들
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "_last_ai_answer" not in st.session_state:
    st.session_state["_last_ai_answer"] = ""

#streamlit이 다시 시작되도 세션은 유지된다.
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages=True
    )
#파일이 리셋되면 다시 대화를 리셋하기 위해.
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None

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
        # 매 요청마다 초기화 (안 하면 이전 답변 뒤에 계속 붙음)
        self.message = ""
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        if self.message_box is not None:
            self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        # 최종 답변 저장
        save_message(self.message, role="ai")
        st.session_state["_last_ai_answer"] = self.message


#메모리에서 데이터를 가지고 옴. 랭체인 병렬 실행으로 인해 get_memory에서 memory를 초기화해서 만들어줌.
def get_memory():
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    return st.session_state["memory"]

def load_chat_history(_):
    memory = get_memory()
    return memory.load_memory_variables({})["chat_history"]

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_bytes: bytes, file_name: str, api_key: str):
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

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------
# Prompt (키 일치!)
# ---------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n"
            "Use BOTH the uploaded file context and the chat history to answer.\n"
            "If you don't know, say you don't know. Don't make anything up.\n\n"
            "Context:\n{context}\n\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------
# UI
# ---------------------------
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

#Main
if file and api_key:
    #파일이 전환되면 많은 데이터 리셋
    if st.session_state.get("active_file") != file.name:
        st.session_state["active_file"] = file.name
        st.session_state["messages"] = []
        st.session_state["_last_ai_answer"] =""
        st.session_state.pop("memory", None)
        _ = get_memory() #새 메모리 생성
        
        
    file_bytes = file.getvalue()
    retriever = embed_file(file_bytes, file.name, api_key)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        openai_api_key=api_key,
        callbacks=[ChatCallbackHandler()],
    )

    # 체인은 한번만 만들고 재사용
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "chat_history": RunnableLambda(load_chat_history), 
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    user_msg = st.chat_input("Ask anything about your file...")
    if user_msg:
        send_message(user_msg, "human")
        #메시지를 하나씩 저장해야 참조할 수 있음.
        memory = get_memory()
        #memory는 st.session_state["messages"]로 대용할 수 있음.
        memory.chat_memory.add_user_message(user_msg)
        # 실행 + memory 저장
        with st.chat_message("ai"):
            _ = chain.invoke(user_msg)  # result는 AIMessage
        #_last_ai_answer가 없으면 ""를 반환
        ai_answer = st.session_state.get("_last_ai_answer", "")
        if ai_answer:
            memory.chat_memory.add_ai_message(ai_answer)

else:
    # 파일/키가 없을 때도 messages를 매번 초기화하면 대화가 계속 날아감.
    # 여기서는 초기화하지 않음.
    pass
