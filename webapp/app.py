import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from search import LawSearch

app = FastAPI(title="Law Library Q&A")
searcher = LawSearch()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class QuestionRequest(BaseModel):
    question: str

class SetKeyRequest(BaseModel):
    key: str

@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/api/docs")
def list_docs():
    return {"documents": searcher.list_docs()}

@app.post("/api/ask")
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Search relevant law sections
    results = searcher.search(req.question, top_k=3)

    if not results:
        return {
            "answer": "ไม่พบข้อมูลกฎหมายที่เกี่ยวข้องในคลังข้อมูล\nNo relevant law found in the library.\n未在法律资料库中找到相关法律信息。",
            "sources": [],
        }

    # Build context from top results
    context_parts = []
    for r in results:
        context_parts.append(f"=== {r['title']} / {r['title_en']} ===\n{r['excerpt']}")
    context = "\n\n".join(context_parts)

    sources = [{"title": r["title"], "title_en": r["title_en"], "path": r["path"]} for r in results]

    if not GEMINI_API_KEY:
        # No API key — return the raw excerpts as answer
        answer = "⚠️ ยังไม่ได้ตั้งค่า Gemini API Key — แสดงข้อความจากกฎหมายที่เกี่ยวข้องโดยตรง\n\n"
        answer += context
        return {"answer": answer, "sources": sources}

    # Call Gemini
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are a Thai law expert who is fluent in Thai, English, and Chinese (Simplified).

Instructions:
1. Detect the language of the question below (Thai / English / Chinese).
2. Answer ONLY based on the provided law excerpts — do not invent information.
3. Respond primarily in the SAME language as the question. Also include a brief summary in the other two languages at the end.
4. Always cite the law name and relevant section/clause.
5. If the question is in Chinese, use Simplified Chinese (简体中文).

คุณเป็นผู้เชี่ยวชาญด้านกฎหมายไทยที่ใช้ภาษาไทย อังกฤษ และจีน (ตัวย่อ) ได้คล่อง
ตอบในภาษาเดียวกับคำถาม และสรุปสั้นๆ เป็นอีกสองภาษาด้านล่าง ระบุชื่อกฎหมายและมาตราที่อ้างอิงเสมอ

--- Law Excerpts / ข้อมูลกฎหมาย / 法律条文 ---
{context}

--- Question / คำถาม / 问题 ---
{req.question}

--- Answer / คำตอบ / 回答 ---"""

        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {str(e)}\n\nข้อมูลจากกฎหมายที่เกี่ยวข้อง:\n\n{context}"

    return {"answer": answer, "sources": sources}

@app.post("/api/set-key")
def set_key(req: SetKeyRequest):
    global GEMINI_API_KEY
    GEMINI_API_KEY = req.key.strip()
    return {"status": "ok"}

@app.post("/api/reload")
def reload_docs():
    searcher.reload()
    return {"status": "ok", "doc_count": len(searcher.docs)}

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
