# backend/main.py
import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader

import chromadb
from chromadb.config import Settings

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ----------------------------
# Config (env)
# ----------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "insurtech_lab8")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# CORS for Next.js (local + docker)
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000",
).split(",")


# ----------------------------
# App init
# ----------------------------
app = FastAPI(
    title="InsurTech RAG Underwriter (Lab 8)",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# LLM + Embeddings (Ollama)
# ----------------------------
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# ----------------------------
# Chroma client / vectorstore
# ----------------------------
def get_chroma_client() -> chromadb.HttpClient:
    """
    Robust client creation:
    - In docker, Chroma may not be ready when API starts.
    - Retry a few times to avoid "Could not connect" during first ingest.
    """
    last_err: Optional[Exception] = None
    for _ in range(20):
        try:
            client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(anonymized_telemetry=False),
            )
            client.list_collections()  # probe
            return client
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5)

    raise HTTPException(
        status_code=503,
        detail=f"Chroma is not reachable at {CHROMA_HOST}:{CHROMA_PORT}. Error: {last_err}",
    )


def get_vectorstore() -> Chroma:
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "•", "-", " ", ""],
)


# ----------------------------
# Prompt (LLM draft for teaching)
# ----------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "คุณเป็น AI Underwriter ที่ตัดสินใจแบบ evidence-based.\n"
            "ข้อบังคับ:\n"
            "1) ใช้ข้อมูลจาก Context เท่านั้น ห้ามเดา/เติมเอง\n"
            "2) ก่อนตัดสิน ต้องระบุ 'เกณฑ์รับประกันที่พบในเอกสาร' และ 'ค่าจริงของผู้เอาประกัน' พร้อม [S#]\n"
            "3) ถ้าไม่พบเกณฑ์ หรือไม่พบค่าจริง (เช่น BP/BMI) ให้ตอบว่า 'ต้องส่งพิจารณาเพิ่ม' และบอกว่าขาดอะไร\n"
            "4) ทุก bullet ในเหตุผลต้องมี [S#] เสมอ\n"
            "5) ถ้า Context มีคำว่า 'แข่งรถ' ให้ถือว่าเป็นความเสี่ยง ต้องตั้งคำถามเพิ่มเสมอ\n"
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "คำถาม: {question}\n\n"
            "คำตอบต้องมีโครงสร้าง:\n"
            "1) ผลการพิจารณา: ผ่าน/ไม่ผ่าน/ต้องส่งพิจารณาเพิ่ม\n"
            "2) เหตุผลแบบ bullet (ทุก bullet ต้องใส่ [S#])\n"
            "3) ความเสี่ยง/ประเด็นที่ต้องถามเพิ่ม (ถ้ามี)\n",
        ),
    ]
)


# ----------------------------
# Schemas
# ----------------------------
class AnalyzeRequest(BaseModel):
    question: str


# ----------------------------
# Utilities
# ----------------------------
def read_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Return list of {page, text}."""
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def read_text_bytes(raw: bytes) -> str:
    # Try utf-8; fallback without crashing
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def build_docs(
    source: str,
    text: str,
    page: Optional[int] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    chunks = splitter.split_text(text)
    docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        meta: Dict[str, Any] = {"source": source, "chunk": idx}
        if extra_meta:
            meta.update(extra_meta)
        if page is not None:
            meta["page"] = page
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("chunk"),
            hashlib.sha1(d.page_content.encode("utf-8", errors="ignore")).hexdigest(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def format_context_with_sources(docs: List[Document]) -> str:
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        chunk = d.metadata.get("chunk", None)
        doc_type = d.metadata.get("doc_type", None)

        meta = f"source={src}"
        if doc_type:
            meta += f", doc_type={doc_type}"
        if page is not None:
            meta += f", page={page}"
        if chunk is not None:
            meta += f", chunk={chunk}"

        lines.append(f"[S{i}] ({meta})\n{d.page_content}")
    return "\n\n".join(lines)


def _extract_policy_thresholds(policy_text: str) -> Tuple[Optional[Tuple[int, int]], Optional[float], Optional[float]]:
    """
    Return: (threshold_bp_max, bmi_min, bmi_max)
    Accept Thai phrasings like:
    - "ไม่เกิน 140/90 mmHg"
    - "สูงเกิน 140/90 mmHg" (interpreted as max=140/90)
    - "BMI ไม่เกิน 27.5"
    - "BMI อยู่ระหว่าง 18.5 ถึง 27.5"
    """
    policy_l = policy_text.lower()

    thr_bp: Optional[Tuple[int, int]] = None
    bp_patterns = [
        r"ไม่เกิน\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?",
        r"(?:สูง)?เกิน\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?",
    ]
    for pat in bp_patterns:
        m = re.search(pat, policy_l, flags=re.I)
        if m:
            thr_bp = (int(m.group(1)), int(m.group(2)))
            break

    bmi_min: Optional[float] = None
    bmi_max: Optional[float] = None

    m = re.search(
        r"bmi\s*(?:อยู่ระหว่าง|อยู่ในช่วง)\s*(\d+(?:\.\d+)?)\s*(?:ถึง|-|–)\s*(\d+(?:\.\d+)?)",
        policy_l,
        flags=re.I,
    )
    if m:
        bmi_min = float(m.group(1))
        bmi_max = float(m.group(2))

    m2 = re.search(r"bmi\s*ไม่เกิน\s*(\d+(?:\.\d+)?)", policy_l, flags=re.I)
    if m2:
        bmi_max = float(m2.group(1))

    return thr_bp, bmi_min, bmi_max


def _extract_health_values(health_text: str) -> Tuple[Optional[Tuple[int, int]], Optional[float], bool]:
    health_l = health_text.lower()

    act_bp: Optional[Tuple[int, int]] = None
    m = re.search(r"ความดัน[^\d]*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?", health_l, flags=re.I)
    if m:
        act_bp = (int(m.group(1)), int(m.group(2)))

    act_bmi: Optional[float] = None
    m2 = re.search(r"bmi[^\d]*(\d+(?:\.\d+)?)", health_l, flags=re.I)
    if m2:
        act_bmi = float(m2.group(1))

    exclusion = False
    if "แข่งรถ" in health_text or "gokart" in health_l or "go-kart" in health_l:
        exclusion = True

    return act_bp, act_bmi, exclusion


def simple_rule_check(context: Union[str, List[Document]]) -> Dict[str, Any]:
    """
    Classroom audit gate.
    - If context is List[Document], separate policy/health by doc_type.
    - If context is str, treat it as blended context (less reliable).
    """
    if isinstance(context, list):
        policy_text = "\n".join([d.page_content for d in context if d.metadata.get("doc_type") == "policy"])
        health_text = "\n".join([d.page_content for d in context if d.metadata.get("doc_type") == "health"])
    else:
        policy_text = context
        health_text = context

    thr_bp, thr_bmi_min, thr_bmi_max = _extract_policy_thresholds(policy_text)
    act_bp, act_bmi, exclusion = _extract_health_values(health_text)

    decision = "ต้องส่งพิจารณาเพิ่ม"
    reasons: List[str] = []

    missing: List[str] = []
    if thr_bp is None:
        missing.append("ไม่พบเกณฑ์ความดันใน policy (เช่น 'ไม่เกิน 140/90 mmHg' หรือ 'สูงเกิน 140/90 mmHg')")
    if thr_bmi_max is None:
        missing.append("ไม่พบเกณฑ์ BMI ใน policy (เช่น 'BMI ไม่เกิน 27.5' หรือ 'BMI อยู่ระหว่าง 18.5 ถึง 27.5')")
    if act_bp is None:
        missing.append("ไม่พบค่าความดันของผู้เอาประกันในรายงานสุขภาพ")
    if act_bmi is None:
        missing.append("ไม่พบค่า BMI ของผู้เอาประกันในรายงานสุขภาพ")

    if missing:
        reasons.extend(missing)
        decision = "ต้องส่งพิจารณาเพิ่ม"
    else:
        if act_bp[0] > thr_bp[0] or act_bp[1] > thr_bp[1]:
            reasons.append(f"ความดัน {act_bp[0]}/{act_bp[1]} สูงกว่าเกณฑ์ {thr_bp[0]}/{thr_bp[1]}")
        if thr_bmi_max is not None and act_bmi > thr_bmi_max:
            reasons.append(f"BMI {act_bmi:.1f} สูงกว่าเกณฑ์ {thr_bmi_max:.1f}")

        decision = "ไม่ผ่าน" if reasons else "ผ่าน"

        if exclusion:
            if decision == "ผ่าน":
                decision = "ต้องส่งพิจารณาเพิ่ม"
            reasons.append('พบงานอดิเรก/กิจกรรมเข้าข่าย "แข่งรถ" (ต้องตีความข้อยกเว้น/ขอข้อมูลเพิ่ม)')

    return {
        "decision": decision,
        "reasons": reasons,
        "extracted": {
            "threshold_bp": list(thr_bp) if thr_bp else None,
            "threshold_bmi_min": thr_bmi_min,
            "threshold_bmi_max": thr_bmi_max,
            "actual_bp": list(act_bp) if act_bp else None,
            "actual_bmi": act_bmi,
            "exclusion_flag": exclusion,
        },
        "note": "rule_check เป็นกฎง่าย ๆ เพื่อใช้ตรวจคำตอบ (audit) ไม่ใช่ระบบ underwriting จริง",
    }


def validate_llm_citations(text: str) -> List[str]:
    """Every bullet line must contain [S#]."""
    violations: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("-") and re.search(r"\[S\d+\]", s) is None:
            violations.append(f"Bullet ไม่มีการอ้างอิง [S#]: {s[:120]}")
    return violations


def retrieve_two_stage(vs: Chroma, question: str) -> Dict[str, List[Document]]:
    """
    Two-stage retrieval with query hints to avoid pulling only 'signature/claims' chunks.
    """
    policy_hint = (
        question
        + "\nเกณฑ์รับประกันสุขภาพ ความดัน 140/90 mmHg BMI 27.5 ข้อยกเว้น กิจกรรมเสี่ยง แข่งรถ โกคาร์ท"
    )
    health_hint = question + "\nรายงานสุขภาพ ความดัน BMI งานอดิเรก แข่งรถ โกคาร์ท"

    policy_docs = vs.similarity_search(policy_hint, k=5, filter={"doc_type": "policy"})
    health_docs = vs.similarity_search(health_hint, k=4, filter={"doc_type": "health"})

    policy_criteria_docs = vs.similarity_search(
        "เกณฑ์รับประกันสุขภาพ ไม่เกิน 140/90 mmHg BMI อยู่ระหว่าง 18.5 ถึง 27.5 ข้อยกเว้น แข่งรถ",
        k=4,
        filter={"doc_type": "policy"},
    )

    policy_docs = dedup_docs(policy_docs + policy_criteria_docs)
    health_docs = dedup_docs(health_docs)

    return {"policy_docs": policy_docs, "health_docs": health_docs}


# ----------------------------
# Routes
# ----------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "collection": CHROMA_COLLECTION,
    }


@app.post("/reset")
def reset_collection():
    """Reset collection for classroom reuse."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=CHROMA_COLLECTION)
    except Exception:
        pass
    client.get_or_create_collection(name=CHROMA_COLLECTION)
    return {"status": "reset", "collection": CHROMA_COLLECTION}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest TXT/PDF into Chroma."""
    filename = file.filename or "uploaded"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = (filename.split(".")[-1] if "." in filename else "").lower()
    docs: List[Document] = []

    fn_l = filename.lower()
    if ("policy" in fn_l) or ("กรมธรรม์" in filename):
        doc_type = "policy"
    elif ("health" in fn_l) or ("report" in fn_l) or ("รายงานสุขภาพ" in filename):
        doc_type = "health"
    else:
        doc_type = "other"

    if ext == "pdf":
        pages = read_pdf_bytes(raw)
        for p in pages:
            text = (p["text"] or "").strip()
            if not text:
                continue
            docs.extend(build_docs(source=filename, text=text, page=p["page"], extra_meta={"doc_type": doc_type}))
    else:
        text = read_text_bytes(raw).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text extracted (file may be empty or not text-based)")
        docs = build_docs(source=filename, text=text, extra_meta={"doc_type": doc_type})

    if not docs:
        raise HTTPException(status_code=400, detail="No ingestable text found (PDF may be scanned image)")

    vs = get_vectorstore()
    vs.add_documents(docs)

    return {
        "status": "ingested",
        "source": filename,
        "doc_type": doc_type,
        "chunks_added": len(docs),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection": CHROMA_COLLECTION,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    vs = get_vectorstore()

    retrieved = retrieve_two_stage(vs, question)
    policy_docs = retrieved["policy_docs"]
    health_docs = retrieved["health_docs"]
    docs = dedup_docs(policy_docs + health_docs)

    if not docs:
        msg = "ต้องส่งพิจารณาเพิ่ม: ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ (ยังไม่ได้ ingest หรือข้อมูลไม่ตรงคำถาม)"
        return {
            "final_answer": msg,
            "answer": msg,  # UI compatibility
            "llm_draft": "",
            "sources": [],
            "rule_check": {"decision": "ต้องส่งพิจารณาเพิ่ม", "reasons": ["ไม่พบหลักฐานจาก retrieval"]},
            "audit_alerts": ["ไม่พบหลักฐานจาก retrieval — ตรวจว่า ingest แล้วหรือยัง"],
        }

    if len(policy_docs) == 0 or len(health_docs) == 0:
        missing: List[str] = []
        if len(policy_docs) == 0:
            missing.append("ไม่พบหลักฐานฝั่ง policy (เกณฑ์/ข้อยกเว้น) — ตั้งชื่อไฟล์ให้มีคำว่า policy และ ingest policy ให้ถูก")
        if len(health_docs) == 0:
            missing.append("ไม่พบหลักฐานฝั่ง health/report (ค่าจริง/พฤติกรรม) — ตั้งชื่อไฟล์ให้มีคำว่า health/report และ ingest health report ให้ถูก")
        msg = "ต้องส่งพิจารณาเพิ่ม: หลักฐานไม่ครบสองฝั่ง (policy + health report)"
        return {
            "final_answer": msg,
            "answer": msg,
            "llm_draft": "",
            "sources": [],
            "rule_check": {"decision": "ต้องส่งพิจารณาเพิ่ม", "reasons": missing},
            "audit_alerts": ["evidence incompleteness — retrieval ไม่ครบเอกสารสองประเภท"],
        }

    context = format_context_with_sources(docs)
    msg = PROMPT.format_messages(context=context, question=question)
    resp = llm.invoke(msg)

    rule = simple_rule_check(docs)

    audit_alerts: List[str] = []
    if rule.get("decision") == "ต้องส่งพิจารณาเพิ่ม":
        audit_alerts.append("หลักฐานไม่พอสำหรับตัดสิน (missing criteria/values หรือมีความเสี่ยงต้องถามเพิ่ม)")

    citation_violations = validate_llm_citations(resp.content or "")
    if citation_violations:
        audit_alerts.append("LLM draft ไม่ผ่าน citation policy — จะไม่ใช้เป็นผลสรุป")
        audit_alerts.extend(citation_violations[:3])

    sources: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, start=1):
        preview = d.page_content
        if len(preview) > 180:
            preview = preview[:180] + "..."
        sources.append(
            {
                "sid": f"S{i}",
                "source": d.metadata.get("source"),
                "doc_type": d.metadata.get("doc_type"),
                "page": d.metadata.get("page"),
                "chunk": d.metadata.get("chunk"),
                "preview": preview,
            }
        )

    final_answer = resp.content or ""
    if (rule.get("decision") in ["ต้องส่งพิจารณาเพิ่ม", "ไม่ผ่าน"]) or citation_violations:
        lines = [f"1) ผลการพิจารณา: {rule.get('decision')}"]
        rs = rule.get("reasons") or []
        if rs:
            lines.append("")
            lines.append("2) เหตุผล (Audit Gate):")
            for r in rs:
                lines.append(f"- {r}")
        lines.append("")
        lines.append("3) หมายเหตุ: ผลนี้มาจาก rule_check/citation gate เพื่อกันการตัดสินเมื่อหลักฐานไม่ครบหรือมีข้อยกเว้น")
        final_answer = "\n".join(lines)

    return {
        "final_answer": final_answer,
        "answer": final_answer,
        "llm_draft": resp.content,
        "sources": sources,
        "rule_check": rule,
        "audit_alerts": audit_alerts,
    }
