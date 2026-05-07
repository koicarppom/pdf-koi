import os
import streamlit as st
from dotenv import load_dotenv
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")
st.caption("อัปโหลด PDF แล้วถามคำถามได้เลยค่ะ!")

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_answer(vectorstore, question, chat_history):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    prompt = f"""คุณเป็น AI assistant ที่ตอบคำถามจากเอกสาร PDF

เนื้อหาจาก PDF:
{context}

ประวัติการสนทนา:
{history_text}

คำถาม: {question}

ตอบคำถามโดยอิงจากเนื้อหาใน PDF เท่านั้น ถ้าไม่มีข้อมูลให้บอกว่าไม่พบในเอกสารค่ะ"""

    response = llm.invoke(prompt)
    return response.content

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("⏳ กำลังอ่านและประมวลผล PDF..."):
        text = read_pdf(uploaded_file)
        st.session_state.vectorstore = create_vectorstore(text)
    st.success("✅ พร้อมแล้ว! ถามคำถามได้เลยค่ะ")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("ถามคำถามเกี่ยวกับ PDF..."):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ กรุณาอัปโหลด PDF ก่อนนะคะ")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิด..."):
                answer = get_answer(
                    st.session_state.vectorstore,
                    prompt,
                    st.session_state.messages
                )
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
