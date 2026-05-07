import os
import streamlit as st
from dotenv import load_dotenv
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF Chatbot")
st.caption("อัปโหลด PDF แล้วถามคำถามได้เลยค่ะ!")

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore(text):
    # chunk ใหญ่ขึ้น + overlap มากขึ้น = ค้นหาได้แม่นยำขึ้น
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
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
        temperature=0.2
    )

    # ดึง context จาก PDF มากขึ้น
    docs = vectorstore.similarity_search(question, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # สร้าง history สำหรับ context
    history_text = ""
    for msg in chat_history[-6:]:  # จำ 6 ข้อความล่าสุด
        role = "ผู้ใช้" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""คุณเป็น AI assistant ผู้เชี่ยวชาญในการวิเคราะห์เอกสาร PDF
คุณต้องตอบคำถามโดยอิงจากเนื้อหาใน PDF เท่านั้น

กฎสำคัญ:
1. ตอบเป็นภาษาเดียวกับคำถาม (ถามไทย ตอบไทย / ถามอังกฤษ ตอบอังกฤษ)
2. ถ้าข้อมูลอยู่ใน PDF ให้ตอบแบบละเอียดและชัดเจน
3. ถ้าไม่มีข้อมูลใน PDF ให้บอกตรงๆ ว่า "ไม่พบข้อมูลนี้ในเอกสารค่ะ"
4. จำบทสนทนาก่อนหน้าและใช้ context ต่อเนื่องได้
5. ถ้าคำถามเกี่ยวกับบทสนทนาก่อนหน้า ให้ใช้ประวัติการสนทนาด้วย

เนื้อหาจาก PDF:
{context}

ประวัติการสนทนา:
{history_text}

คำถามปัจจุบัน: {question}

คำตอบ:"""

    response = llm.invoke(prompt)
    return response.content

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Upload PDF
uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type="pdf")

if uploaded_file:
    if st.session_state.pdf_name != uploaded_file.name:
        with st.spinner("⏳ กำลังอ่านและประมวลผล PDF..."):
            text = read_pdf(uploaded_file)
            if len(text.strip()) < 100:
                st.error("❌ ไม่สามารถอ่านข้อความจาก PDF ได้ค่ะ กรุณาลอง PDF อื่น")
            else:
                st.session_state.vectorstore = create_vectorstore(text)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.success(f"✅ โหลด '{uploaded_file.name}' สำเร็จแล้วค่ะ! ({len(text)} ตัวอักษร)")

# แสดงจำนวน message
if st.session_state.messages:
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("🗑️ ล้างประวัติ"):
            st.session_state.messages = []
            st.rerun()

# แสดง chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("ถามคำถามเกี่ยวกับ PDF..."):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ กรุณาอัปโหลด PDF ก่อนนะคะ")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังค้นหาและวิเคราะห์..."):
                answer = get_answer(
                    st.session_state.vectorstore,
                    prompt,
                    st.session_state.messages
                )
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )