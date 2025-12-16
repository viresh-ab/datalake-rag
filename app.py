import os
import json
import requests
import streamlit as st
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# =====================================================
# BASIC SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SITE_ID = os.getenv("SITE_ID")

EMBED_MODEL = "text-embedding-3-large"   # 3072 dims
CHAT_MODEL = "gpt-4.1-mini"

INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
META_PATH = os.path.join(FAISS_DIR, "meta.pkl")

GRAPH = "https://graph.microsoft.com/v1.0"

client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# MICROSOFT AUTH
# =====================================================
def get_access_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default"
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def graph_headers():
    return {"Authorization": f"Bearer {get_access_token()}"}

# =====================================================
# SHAREPOINT DISCOVERY (ID-BASED – SAFE)
# =====================================================
def find_metadata_file():
    """
    Finds metadata.json across all document libraries
    using item ID (no dependency on parentReference.path)
    """
    drives_url = f"{GRAPH}/sites/{SITE_ID}/drives"
    drives_res = requests.get(drives_url, headers=graph_headers())
    drives_res.raise_for_status()

    for drive in drives_res.json()["value"]:
        drive_id = drive["id"]

        search_url = f"{GRAPH}/drives/{drive_id}/root/search(q='metadata.json')"
        search_res = requests.get(search_url, headers=graph_headers())

        if search_res.status_code != 200:
            continue

        for item in search_res.json().get("value", []):
            if item["name"].lower() == "metadata.json":
                return drive_id, item["id"]

    raise RuntimeError("❌ metadata.json not found in any document library")

def download_file_by_id(drive_id, item_id):
    url = f"{GRAPH}/drives/{drive_id}/items/{item_id}/content"
    r = requests.get(url, headers=graph_headers())
    r.raise_for_status()
    return r.content

def get_item_id_by_path(drive_id, file_path):
    """
    Converts a file path from metadata.json into a Graph item ID
    """
    url = f"{GRAPH}/drives/{drive_id}/root:{file_path}"
    r = requests.get(url, headers=graph_headers())
    r.raise_for_status()
    return r.json()["id"]

# =====================================================
# TEXT CHUNKING
# =====================================================
def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# =====================================================
# INGESTION
# =====================================================
def run_ingestion():
    st.info("🔍 Locating metadata.json in SharePoint...")

    drive_id, metadata_item_id = find_metadata_file()
    st.success("✅ Found metadata.json")

    metadata = json.loads(download_file_by_id(drive_id, metadata_item_id))

    if "Case Study" not in metadata or not isinstance(metadata["Case Study"], list):
        st.error("❌ metadata.json must contain 'Case Study' as a list")
        return

    index = faiss.IndexFlatL2(3072)
    meta_store = []

    for item in metadata["Case Study"]:
        file_path = item.get("file_path")
        if not file_path:
            continue

        try:
            pdf_item_id = get_item_id_by_path(drive_id, file_path)
            pdf_bytes = download_file_by_id(drive_id, pdf_item_id)
        except Exception as e:
            st.warning(f"⚠️ Skipped file: {file_path}")
            continue

        temp_pdf = os.path.join(TEMP_DIR, os.path.basename(file_path))
        with open(temp_pdf, "wb") as f:
            f.write(pdf_bytes)

        reader = PdfReader(temp_pdf)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        os.remove(temp_pdf)

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk
            ).data[0].embedding

            index.add(np.array([emb]).astype("float32"))
            meta_store.append({
                **item,
                "chunk_id": i,
                "text": chunk
            })

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_store, f)

    st.success("🚀 Ingestion completed successfully")

# =====================================================
# LOAD VECTOR STORE
# =====================================================
@st.cache_resource
def load_store():
    if not os.path.exists(INDEX_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return index, meta

# =====================================================
# ASK QUESTION (RAG)
# =====================================================
def ask(question):
    index, meta = load_store()

    if index is None:
        return "❌ Data not ingested yet. Click **Run Ingestion** first."

    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=question
    ).data[0].embedding

    _, I = index.search(np.array([q_emb]).astype("float32"), 5)
    context = "\n\n".join([meta[i]["text"] for i in I[0]])

    response = client.responses.create(
        model=CHAT_MODEL,
        input=f"""
Use ONLY the context below.

Context:
{context}

Question:
{question}
"""
    )

    return response.output_text

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(
    page_title="SharePoint Data Lake AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SharePoint Data Lake – AI Assistant")
st.caption("SharePoint RAG | ID-safe Graph | FAISS | OpenAI")

with st.sidebar:
    st.header("⚙️ Admin")
    if st.button("🚀 Run Ingestion", key="run_ingestion_btn"):
        run_ingestion()

st.divider()

question = st.text_area(
    "Ask a question",
    placeholder="e.g. Create a blog from 2023 Chocolate Taste Preference study",
    height=120
)

if st.button("🔍 Generate Answer", key="generate_answer_btn"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            answer = ask(question)

        st.subheader("✅ Answer")
        st.write(answer)

