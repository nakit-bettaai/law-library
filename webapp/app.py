import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from search import LawSearch

app = FastAPI(title="Law Library Q&A")
searcher = LawSearch()

# API key loaded from environment (set as HF Space secret — never in source code)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"   # 70B, excellent Thai, free 14,400 req/day

SYSTEM_PROMPT = """คุณคือผู้เชี่ยวชาญด้านกฎหมายไทยที่ใช้ภาษาไทย อังกฤษ และจีน (ตัวย่อ) ได้คล่อง
คุณอธิบายกฎหมายในภาษาที่เข้าใจง่าย เหมาะสำหรับคนทั่วไปที่ไม่ใช่นักกฎหมาย

คำแนะนำสำคัญ:
1. ตรวจจับภาษาของคำถาม (ไทย / อังกฤษ / จีน) แล้วตอบในภาษาเดียวกัน
2. ตอบโดยอิงจากข้อมูลกฎหมายที่ให้ไว้เท่านั้น ห้ามสร้างข้อมูลขึ้นเอง
3. **ต้องอ้างอิงข้อความจากมาตราโดยตรง** — คัดลอกข้อความของมาตราที่เกี่ยวข้องในรูปแบบ blockquote (> ...) ก่อนอธิบาย
4. ระบุชื่อกฎหมายและมาตราให้ชัดเจน เช่น **พ.ร.บ.คุ้มครองแรงงาน มาตรา 118**
5. หลังจากอ้างข้อความกฎหมายแล้ว ให้อธิบายความหมายด้วยภาษาที่เข้าใจง่าย
6. จัดรูปแบบให้อ่านง่ายด้วยหัวข้อ bullet points และย่อหน้า
7. สรุปสั้นๆ เป็นอีกสองภาษาต่อท้าย

รูปแบบคำตอบที่ต้องการ:
### [ชื่อกฎหมาย]
**มาตรา [X] — [หัวข้อ]**
> [ข้อความของมาตรานั้นโดยตรงจากแหล่งที่มา]

**ความหมาย:** [อธิบายด้วยภาษาง่าย]

---
You are a Thai law expert fluent in Thai, English, and Simplified Chinese.
Explain laws in plain language for non-lawyers. Answer ONLY based on the provided excerpts.

CRITICAL: Always quote the exact text of relevant provisions verbatim in a blockquote (> ...) before explaining them in plain language. Format:

### [Law Name]
**Section [X] — [Title]**
> [Exact provision text copied from the source]

**Plain meaning:** [Explanation in simple language]

Detect the question language and respond in that language. End with brief summaries in the other two languages."""

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/api/status")
def status():
    return {"ready": bool(GROQ_API_KEY), "model": GROQ_MODEL}

@app.get("/api/docs")
def list_docs():
    return {"documents": searcher.list_docs()}

@app.post("/api/ask")
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    results = searcher.search(req.question, top_k=3)
    if not results:
        return {
            "answer": "ไม่พบข้อมูลกฎหมายที่เกี่ยวข้องในคลังข้อมูล\nNo relevant law found in the library.\n未在法律资料库中找到相关法律信息。",
            "sources": [],
        }

    context = "\n\n".join(
        f"=== {r['title']} / {r['title_en']} ===\n{r['excerpt']}"
        for r in results
    )
    sources = [{"title": r["title"], "title_en": r["title_en"], "path": r["path"]} for r in results]

    if not GROQ_API_KEY:
        return {
            "answer": "⚠️ ระบบ AI ยังไม่พร้อม กรุณาติดต่อผู้ดูแลระบบ\nAI not configured. Please contact the administrator.\n\n" + context,
            "sources": sources,
        }

    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"--- ข้อมูลกฎหมาย / Law Excerpts ---\n{context}\n\n--- คำถาม ---\n{req.question}\n\n--- คำตอบ ---"},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ เกิดข้อผิดพลาด: {str(e)}\n\nข้อมูลจากกฎหมายที่เกี่ยวข้อง:\n\n{context}"

    return {"answer": answer, "sources": sources}

@app.post("/api/reload")
def reload_docs():
    searcher.reload()
    return {"status": "ok", "doc_count": len(searcher.docs)}

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
