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
    tokens = []
    # Split into runs: Thai, English/digits, Chinese
    for chunk in re.finditer(r'[฀-๿]+|[a-z0-9]+|[一-鿿]+', text):
        s = chunk.group()
        if re.match(r'[฀-๿]', s):
            # Thai: use overlapping bigrams so partial words still match
            tokens.extend(s[i:i+2] for i in range(len(s) - 1))
            tokens.extend(s[i:i+3] for i in range(len(s) - 2))
        elif re.match(r'[一-鿿]', s):
            # Chinese: individual characters
            tokens.extend(s)
        else:
            tokens.append(s)
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
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            if score > 0:
                d = self.docs[idx]
                # Top result gets larger excerpt so model can quote provisions verbatim
                max_len = 2000 if rank == 0 else 1000
                excerpt = extract_excerpt(d["content"], tokens, max_len=max_len)
                results.append({
                    "title": d["title"],
                    "title_en": d["title_en"],
                    "path": d["path"],
                    "score": round(score, 3),
                    "excerpt": excerpt,
                })
        return results

    def search_deka(self, query: str, top_k: int = 2) -> list[dict]:
        """Search only within supreme-court documents and return top matches."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        results = []
        for idx, d in enumerate(self.docs):
            if "supreme-court" not in d["path"]:
                continue
            s = scores[idx]
            if s > 0:
                excerpt = extract_excerpt(d["content"], tokens, max_len=1500)
                results.append({
                    "title": d["title"], "title_en": d["title_en"],
                    "path": d["path"], "score": round(s, 3), "excerpt": excerpt,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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
    best_indices = set(idx for _, idx in scored[:8])

    # Always pull in the nearest section heading above each matched line
    # so case numbers (## คำพิพากษาฎีกาที่ XXXX/YYYY) are never dropped
    heading_indices = set()
    for idx in best_indices:
        for j in range(idx, -1, -1):
            if lines[j].startswith('#'):
                heading_indices.add(j)
                break

    all_indices = sorted(best_indices | heading_indices)
    excerpt_lines = []
    for i in all_indices:
        start = max(0, i - 1)
        end = min(len(lines), i + 4)
        excerpt_lines.extend(lines[start:end])
    excerpt = '\n'.join(dict.fromkeys(excerpt_lines))
    return excerpt[:max_len] if len(excerpt) > max_len else excerpt
