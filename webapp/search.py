import os
import re
import frontmatter
from pathlib import Path
from rank_bm25 import BM25Okapi

VAULT_PATH = Path(__file__).parent.parent / "vault"

def load_documents():
    docs = []
    for md_file in VAULT_PATH.rglob("*.md"):
        if ".obsidian" in str(md_file):
            continue
        try:
            post = frontmatter.load(str(md_file))
            content = post.content
            meta = post.metadata
            title = meta.get("title", md_file.stem)
            title_en = meta.get("title_en", "")
            tags = meta.get("tags", [])
            docs.append({
                "path": str(md_file.relative_to(VAULT_PATH)),
                "title": title,
                "title_en": title_en,
                "tags": tags,
                "content": content,
                "full_text": f"{title} {title_en} {' '.join(tags)} {content}",
            })
        except Exception:
            pass
    return docs

def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Thai words, English words, and individual Chinese characters
    tokens = re.findall(r'[฀-๿]+|[a-z0-9]+|[一-鿿]', text)
    return tokens

class LawSearch:
    def __init__(self):
        self.docs = load_documents()
        corpus = [tokenize(d["full_text"]) for d in self.docs]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                d = self.docs[idx]
                # Extract most relevant excerpt (~600 chars around query terms)
                excerpt = extract_excerpt(d["content"], tokens)
                results.append({
                    "title": d["title"],
                    "title_en": d["title_en"],
                    "path": d["path"],
                    "score": round(score, 3),
                    "excerpt": excerpt,
                    "full_content": d["content"],
                })
        return results

    def reload(self):
        self.__init__()

    def list_docs(self) -> list[dict]:
        return [{"title": d["title"], "title_en": d["title_en"], "path": d["path"], "tags": d["tags"]} for d in self.docs]

def extract_excerpt(content: str, tokens: list[str], max_len: int = 800) -> str:
    lines = content.split('\n')
    scored = []
    for i, line in enumerate(lines):
        line_tokens = tokenize(line)
        hits = sum(1 for t in tokens if t in line_tokens)
        scored.append((hits, i))
    scored.sort(reverse=True)
    best_lines = sorted(set(idx for _, idx in scored[:5]))
    excerpt_lines = []
    for i in best_lines:
        start = max(0, i-1)
        end = min(len(lines), i+3)
        excerpt_lines.extend(lines[start:end])
    excerpt = '\n'.join(dict.fromkeys(excerpt_lines))
    return excerpt[:max_len] if len(excerpt) > max_len else excerpt
