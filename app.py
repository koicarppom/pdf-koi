import os
import streamlit as st
from dotenv import load_dotenv
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ---- Page Config ----
st.set_page_config(
    page_title="PDF Chatbot Pro",
    page_icon="🤖",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; padding: 8px; }
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-text {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .pdf-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    .stat-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2d3150;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        color: #888;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="title-text">🤖 PDF Chatbot Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">อัปโหลด PDF หลายไฟล์แล้วถามคำถามได้เลยค่ะ!</p>', unsafe_allow_html=True)

# ---- Functions ----
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore(texts_dict):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    for filename, text in texts_dict.items():
        chunks = splitter.split_text(text)
        # ใส่ชื่อไฟล์กำกับทุก chunk
        chunks_with_source = [f"[จาก: {filename}]\n{chunk}" for chunk in chunks]
        all_chunks.extend(chunks_with_source)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(all_chunks, embeddings)
    return vectorstore

def get_answer(vectorstore, question, chat_history, pdf_names):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    docs = vectorstore.similarity_search(question, k=6)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    history_text = ""
    for msg in chat_history[-8:]:
        role = "ผู้ใช้" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content']}\n"

    pdf_list = ", ".join(pdf_names)
    prompt = f"""คุณเป็น AI assistant ผู้เชี่ยวชาญวิเคราะห์เอกสาร PDF อัจฉริยะ
เอกสารที่โหลดอยู่: {pdf_list}

กฎสำคัญ:
1. ตอบภาษาเดียวกับคำถาม (ถามไทย=ตอบไทย, ถามอังกฤษ=ตอบอังกฤษ)
2. ถ้ามีข้อมูลใน PDF ให้ตอบละเอียด ชัดเจน และเป็นประโยชน์
3. บอกด้วยว่าข้อมูลมาจากไฟล์ไหน ถ้ามีหลายไฟล์
4. ถ้าไม่มีข้อมูลใน PDF ให้บอกตรงๆ
5. จำบทสนทนาก่อนหน้าและใช้ context ต่อเนื่อง
6. ตอบแบบเป็นมิตรและเป็นประโยชน์

เนื้อหาจาก PDF:
{context}

ประวัติการสนทนา:
{history_text}

คำถาม: {question}

คำตอบ:"""

    response = llm.invoke(prompt)
    return response.content

# ---- Session State ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = {}
if "total_chars" not in st.session_state:
    st.session_state.total_chars = 0

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 📁 จัดการเอกสาร")
    uploaded_files = st.file_uploader(
        "เลือกไฟล์ PDF (หลายไฟล์ได้)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = {f.name: f for f in uploaded_files
                    if f.name not in st.session_state.pdf_texts}
        if new_files:
            with st.spinner("⏳ กำลังประมวลผล PDF..."):
                for filename, file in new_files.items():
                    text = read_pdf(file)
                    if len(text.strip()) > 100:
                        st.session_state.pdf_texts[filename] = text
                        st.session_state.total_chars += len(text)
                if st.session_state.pdf_texts:
                    st.session_state.vectorstore = create_vectorstore(
                        st.session_state.pdf_texts
                    )
            st.success(f"✅ โหลด {len(new_files)} ไฟล์สำเร็จ!")

    # แสดงไฟล์ที่โหลดแล้ว
    if st.session_state.pdf_texts:
        st.markdown("### 📄 ไฟล์ที่โหลดแล้ว")
        for filename in st.session_state.pdf_texts:
            st.markdown(f'<span class="pdf-badge">📄 {filename}</span>',
                       unsafe_allow_html=True)

        st.markdown("---")

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.pdf_texts)}</div>
                <div class="stat-label">ไฟล์</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.messages)}</div>
                <div class="stat-label">ข้อความ</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("🗑️ ล้างแชท", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_clear2:
            if st.button("📤 ล้างไฟล์", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.pdf_texts = {}
                st.session_state.messages = []
                st.session_state.total_chars = 0
                st.rerun()

    st.markdown("---")
    st.markdown("### 💡 ตัวอย่างคำถาม")
    st.markdown("""
    - สรุปเนื้อหาทั้งหมดให้หน่อย
    - ข้อมูลสำคัญในเอกสารมีอะไรบ้าง
    - เปรียบเทียบข้อมูลระหว่างไฟล์
    - อธิบาย [หัวข้อ] ให้ละเอียดขึ้น
    """)

# ---- Main Chat Area ----
if not st.session_state.pdf_texts:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #888;">
        <div style="font-size: 4rem;">📂</div>
        <h3 style="color: #667eea;">เริ่มต้นด้วยการอัปโหลด PDF</h3>
        <p>อัปโหลดไฟล์ PDF ในแถบซ้ายมือได้เลยค่ะ<br>
        รองรับหลายไฟล์พร้อมกัน!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # แสดง chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("💬 ถามคำถามเกี่ยวกับเอกสารของคุณ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 กำลังค้นหาและวิเคราะห์..."):
                answer = get_answer(
                    st.session_state.vectorstore,
                    prompt,
                    st.session_state.messages,
                    list(st.session_state.pdf_texts.keys())
                )
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )