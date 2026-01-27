# Lab 8 — InsurTech RAG Underwriter (Ollama + Chroma + FastAPI + Next.js)

โครงการนี้ทำ “ระบบพิจารณารับประกันภัยแบบ RAG” สำหรับการสอน
- ingest เอกสาร (policy / health report) เข้า Chroma
- วิเคราะห์คำถามด้วย LLM (Ollama) + หลักฐาน [S#]
- มี **Audit Gate** (rule_check + citation gate) กัน hallucination

> หมายเหตุ: rule_check เป็นกฎสาธิตในห้องเรียน ไม่ใช่ underwriting จริง

---

## 1) Start (Docker)

```bash
docker compose up -d
docker ps
```

เปิดใช้งาน:
- API: http://localhost:8000/docs
- UI:  http://localhost:3000
- Chroma debug: http://localhost:8001 (optional)

### จุดที่พลาดกันบ่อย: CHROMA_PORT
ใน docker network, Chroma ฟังที่ **8000** เสมอ
แม้จะ map ออก host เป็น `8001:8000` ก็ยังต้องตั้ง `CHROMA_PORT=8000` ใน service `api`

---

## 2) Reset collection (เริ่มแลบใหม่)

```bash
curl -X POST http://localhost:8000/reset
```

---

## 3) Ingest เอกสารตัวอย่าง

ไฟล์ตัวอย่างอยู่ใน `sample_data/` (มีทั้งชื่อไทยและชื่ออังกฤษ)

```bash
curl -X POST -F "file=@sample_data/policy_fintech_elite.txt" http://localhost:8000/ingest
curl -X POST -F "file=@sample_data/somchai_health_report.txt" http://localhost:8000/ingest
```

> ตัวระบบจะ tag `doc_type` จากชื่อไฟล์:
> - policy / กรมธรรม์ → `policy`
> - health / report / รายงานสุขภาพ → `health`

---

## 4) Analyze (curl)

### 4.1 ตัวอย่าง JSON (UTF-8)

Linux/macOS:
```bash
cat > q.json << 'JSON'
{"question":"นายสมชายควรผ่านการอนุมัติหรือไม่? [S#]"}
JSON

curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json; charset=utf-8" \
  --data-binary "@q.json"
```

### 4.2 PowerShell (Windows) — ถ้าเห็น ????? แปลว่า encoding
ใน PowerShell ให้ตั้ง UTF-8 ก่อน

```powershell
chcp 65001 | Out-Null
$OutputEncoding = [System.Text.UTF8Encoding]::new()

@'
{"question":"นายสมชายควรผ่านการอนุมัติหรือไม่? [S#]"}
'@ | Set-Content -Encoding utf8 q.json

curl.exe -s -X POST http://localhost:8000/analyze `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@q.json"
```

---

## 5) ทำไมบางครั้ง LLM ตอบผิด แต่ final_answer ถูก
API ส่งกลับ 2 ชั้น:
- `llm_draft` = คำตอบจาก LLM (เพื่อเทียบให้เห็น hallucination)
- `final_answer` = คำตอบหลังผ่าน **Audit Gate**
  - rule_check: ดึงเกณฑ์จาก policy + ค่าจริงจาก health report แล้วเทียบ
  - citation gate: bullet ที่ไม่มี [S#] จะถูกจัดว่า “draft ไม่น่าเชื่อถือ”

ดังนั้นถ้า LLM เดาเอง (เช่น บอกว่าไม่แข่งรถ ทั้งที่เอกสารบอกว่าแข่ง) ระบบจะไม่ใช้ผลนั้นเป็น final

---

## 6) ถ้าเจอ Error: Could not connect to Chroma server
เช็ค 3 อย่าง:
1) `docker ps` ต้องเห็น `fintech-chroma` Running
2) ใน `docker-compose.yml` ของ `api` ต้องเป็น `CHROMA_PORT=8000`
3) ใน `api` ใช้ `CHROMA_HOST=chromadb` (ชื่อ service)

---

## 7) โครงสร้างโฟลเดอร์

- `backend/` FastAPI + RAG + Audit Gate
- `frontend/` Next.js UI
- `sample_data/` policy + health report


## วิธีตรวจสอบ/ใช้งาน Chroma:
### ทดสอบว่าทำงานหรือไม่:
```bash
curl http://localhost:8001/api/v1/heartbeat
```

### ดู collections:
```bash
curl http://localhost:8001/api/v1/collections
```