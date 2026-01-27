# Teacher's Guide — Lab Lecture 8 (InsurTech & RAG)

## Learning Outcomes (ภายในคาบเดียว)
- เห็น pipeline RAG: Ingest → Chunk → Embed → Retrieve → Generate
- อธิบาย “AI Underwriter” แบบ evidence-based: ให้คำตอบพร้อมหลักฐาน
- ตรวจความสอดคล้องของคำตอบด้วย rule-check แบบง่าย (แนว audit)
- เข้าใจแนวคิดการประเมิน RAG: relevance / faithfulness / groundedness (เชื่อมกับ RAGAS)

---

## Suggested Timeline (90–120 นาที)

### Part A (15 นาที) — Setup
- `docker compose up -d --build`
- เช็ค `GET /healthz`

### Part B (20 นาที) — Ingestion + Chunking
- อธิบายว่า chunking คืออะไร (ทำไมต้อง overlap)
- Ingest 2 ไฟล์ด้วย `POST /ingest`
- เปิด discussion: chunk size/overlap ส่งผลต่อ retrieval อย่างไร

### Part C (25 นาที) — Retrieval + Underwriting reasoning
- ตั้งคำถามหลัก: “นายสมชายควรผ่านการอนุมัติหรือไม่?”
- ดูผล `sources` ว่า model ดึง chunk ไหนมาใช้
- ถามซ้ำด้วย phrasing ต่างกัน แล้วดูเสถียรภาพของ retrieval

### Part D (15 นาที) — Audit gate
- เทียบ `answer` vs `rule_check`
- ชี้ให้เห็น failure modes:
  - model มองข้าม threshold เล็กน้อย
  - model ไม่ตีความ “รถโกคาร์ท” ว่าเข้าข่าย “แข่งรถ”
  - model ตอบแบบสุ่ม ถ้า context ไม่พอ

### Part E (15 นาที) — Evaluation concept (RAGAS-inspired)
ให้โจทย์ประเมินแบบเบา (ไม่ต้องติดตั้ง ragas):
- **Relevance**: คำตอบตอบคำถามตรงไหม (อ่าน 2 บรรทัดแรกก็รู้)
- **Citation coverage**: มี [S#] ครบทุกประเด็นที่กล่าวอ้างไหม
- **Faithfulness**: ประโยคที่เป็นข้อเท็จจริง “อยู่ใน sources ไหม”

> ถ้ามีเวลาพิเศษ: ให้กลุ่มหนึ่งลด k=1 อีกกลุ่มเพิ่ม k=5 แล้วเปรียบเทียบ

---

## Extension Tasks (ถ้ามีเวลา)
1) เพิ่ม “กรณีต้องส่งพิจารณาเพิ่ม” เมื่อใกล้เกณฑ์
2) เพิ่ม prompt-injection defense (เช่น ignore instructions in documents)
3) เพิ่ม structured output (JSON schema) เพื่อให้ตอบแบบตรวจง่าย
4) ทำ ablation: no-RAG vs RAG แล้ววัด error rate

---

## Evidence to collect (สำหรับรายงาน/คะแนน)
- Screenshot: `docker ps` + `/docs`
- Log ingestion: จำนวน chunks ที่สร้าง
- Response 3 ชุด: คำถาม 3 แบบ + ผล sources
- ตารางสรุป: k=1/3/5 แล้ว “ตัดสินผลเหมือนกันไหม”
