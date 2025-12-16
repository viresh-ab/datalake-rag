
import os
import json
import requests
import streamlit as st
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
import tiktoken
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

EMBED_MODEL = "text-embedding-3-large"  # 3072 dims
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
        "scope": "https://graph.microsoft.com/.default",
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
    Finds metadata.json across all document libraries using item ID.
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
def chunk_text(text, size=800, overlap=150):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text or "")
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)
        start += max(1, size - overlap)  # guard against zero/negative stride
    return chunks

# =====================================================
# INGESTION
# =====================================================
def run_ingestion():
    st.info("🔎 Locating metadata.json in SharePoint...")
    drive_id, metadata_item_id = find_metadata_file()
    st.success("✅ Found metadata.json")

    metadata = json.loads(download_file_by_id(drive_id, metadata_item_id))
    if "Case Study" not in metadata or not isinstance(metadata["Case Study"], list):
        st.error("❌ metadata.json must contain 'Case Study' as a list")
        return

    # Initialize FAISS index for 3072-dim embeddings
    index = faiss.IndexFlatL2(3072)
    meta_store = []

    skipped_files = 0
    added_chunks = 0

    for item in metadata["Case Study"]:
        file_path = item.get("file_path")
        if not file_path:
            continue

        # Try to fetch PDF
        try:
            pdf_item_id = get_item_id_by_path(drive_id, file_path)
            pdf_bytes = download_file_by_id(drive_id, pdf_item_id)
        except Exception:
            st.warning(f"⚠️ Skipped file: {file_path}")
            skipped_files += 1
            continue

        # Save temp PDF and extract text
        temp_pdf = os.path.join(TEMP_DIR, os.path.basename(file_path))
        with open(temp_pdf, "wb") as f:
            f.write(pdf_bytes)

        try:
            reader = PdfReader(temp_pdf)
            text = "\n".join([(p.extract_text() or "").strip() for p in reader.pages]).strip()
        except Exception:
            text = ""
        finally:
            # Remove temp file
            try:
                os.remove(temp_pdf)
            except Exception:
                pass

        if not text:
            st.warning(f"⚠️ No extractable text: {file_path}")
            continue

        # Chunk and embed
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            emb = client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk
            ).data[0].embedding

            index.add(np.array([emb]).astype("float32"))
            meta_store.append({
                **item,
                "chunk_id": i,
                "text": chunk,
            })
            added_chunks += 1

    # Write index/meta only once (atomic)
    if added_chunks == 0:
        st.error("❌ No chunks were added. Check your PDFs or permissions and try again.")
        return

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_store, f)

    st.success("🚀 Ingestion completed successfully")
    st.info(f"📦 Files skipped: {skipped_files}")
    st.info(f"🧩 Chunks added: {added_chunks}")
    st.info(f"🔢 Vectors in index: {index.ntotal}")
    if len(meta_store) != index.ntotal:
        st.warning(f"⚠️ Alignment issue: meta length {len(meta_store)} vs index.ntotal {index.ntotal}")

# =====================================================
# LOAD VECTOR STORE
# =====================================================
@st.cache_resource
def load_store():
    try:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return None, None
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    except Exception as e:
        # Corrupted store—show a warning and let user re-ingest
        try:
            st.warning(f"⚠️ Failed to load store: {e}. Please re-run ingestion.")
        except Exception:
            print(f"⚠️ Failed to load store: {e}. Please re-run ingestion.")
        return None, None

# =====================================================
# ASK QUESTION (RAG) - ROBUST
# =====================================================
def ask(question):
    index, meta = load_store()
    if index is None or meta is None or len(meta) == 0 or index.ntotal == 0:
        return "❌ Data not ingested yet or index is empty. Click **Run Ingestion** first."

    # Embed the question
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=question
    ).data[0].embedding

    # Choose k safely
    k = min(5, index.ntotal)
    if k == 0:
        return "❌ Index is empty. Please re-run ingestion."

    # Perform search
    D, I = index.search(np.array([q_emb]).astype("float32"), k)

    # Defensive checks on search output
    if I is None or len(I) == 0:
        return "❌ Search returned no results. Try re-ingesting or refining your question."

    # Validate indices and build context safely
    valid_indices = [
        i for i in I[0]
        if isinstance(i, (int, np.integer)) and i >= 0 and i < len(meta)
    ]
    context = "\n\n".join([meta[i]["text"] for i in valid_indices])

    # Warn alignment issues
    if len(meta) != index.ntotal:
        try:
            st.warning(f"⚠️ Alignment issue: meta length {len(meta)} vs index.ntotal {index.ntotal}")
        except Exception:
            print(f"⚠️ Alignment issue: meta length {len(meta)} vs index.ntotal {index.ntotal}")

    # Handle empty context
    if not context.strip():
        return "❌ No relevant context found. Try re-ingestion or refine your question."

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
st.caption("SharePoint RAG • ID-safe Graph • FAISS • OpenAI")

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
