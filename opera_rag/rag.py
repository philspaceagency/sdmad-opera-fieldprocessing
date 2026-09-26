"""Question answering over the indexed repositories."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

from .index import HybridIndex
from .loaders import Chunk, load_repo, repo_map, resolve_repo

DEFAULT_REPOS = ["https://github.com/philspaceagency/sdmad-opera-fieldprocessing.git"]
DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a code assistant for PhilSA's OpERA field-processing tooling \
(underwater benthic GoPro video → frames → GPX/echosounder geotagging → YOLO benthic classification).

Answer ONLY from the repository excerpts provided. Rules:
- Cite every claim with the excerpt number in square brackets, e.g. [2] or [1][4].
- Quote exact function names, arguments, file paths, defaults and command lines as they appear in the code.
- When asked how to run something, give the concrete command or call, then explain what each part does.
- If the excerpts don't contain the answer, say so plainly and name the file(s) most likely to hold it \
(the repository map lists every file). Never invent functions, flags or parameters.
- Point out real problems you notice in the quoted code when relevant to the question \
(hard-coded Windows paths, timezone assumptions, missing error handling), briefly.
- Some values show as <REDACTED>: these are credentials removed at indexing time. Never guess them."""


def format_context(hits) -> str:
    blocks = []
    for n, (c, score, _) in enumerate(hits, 1):
        lang = "python" if c.path.endswith((".py", ".ipynb")) else ""
        meta = c.meta or {}
        extra = f"\nNotebook context: {meta['context']}" if meta.get("context") else ""
        blocks.append(f"[{n}] {c.header()}{extra}\n```{lang}\n{c.text}\n```")
    return "\n\n".join(blocks)


class OperaRAG:
    def __init__(self, index: HybridIndex, llm_model: str = DEFAULT_LLM, api_key: str | None = None):
        self.index = index
        self.llm_model = llm_model
        self.history: list[dict] = []
        self._client = None
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key:
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
            except ImportError:
                print("google-genai package not installed → retrieval-only mode (pip install google-genai)")

    # ------------------------------------------------------------ construction
    @classmethod
    def build(cls, repos: list[str] | None = None, embed_model: str | None = DEFAULT_EMBED,
              include_docs: bool = True, workdir: str = "repos", **kw) -> "OperaRAG":
        chunks = []
        for src in repos or DEFAULT_REPOS:
            root = resolve_repo(src, workdir)
            cs, files = load_repo(root, include_docs=include_docs)
            print(f"  {root.name}: {len(files)} files → {len(cs)} chunks")
            chunks += cs
        idx = HybridIndex(chunks, repo_map(chunks), embed_model=embed_model)
        mode = "hybrid (BM25 + dense)" if idx.vectors is not None else "keyword (BM25)"
        print(f"Index ready: {len(chunks)} chunks, {mode} retrieval")
        return cls(idx, **kw)

    def save(self, folder: str | Path):
        self.index.save(folder)

    @classmethod
    def load(cls, folder: str | Path, **kw) -> "OperaRAG":
        return cls(HybridIndex.load(folder), **kw)

    # ------------------------------------------------------------ querying
    def retrieve(self, question: str, k: int = 8, path_filter: str | None = None):
        """Search, then merge split pieces of the same function/cell into one excerpt
        so a long function doesn't crowd out everything else."""
        raw = self.index.search(question, k=k * 3, path_filter=path_filter)
        groups: dict[tuple, list] = {}
        order = []
        for c, s, parts in raw:
            key = (c.repo, c.path, c.kind, c.name.split(" (part ")[0], (c.meta or {}).get("cell"))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((c, s, parts))
        merged = []
        for key in order[:k]:
            items = groups[key]
            if len(items) == 1:
                merged.append(items[0])
                continue
            # keep the best-scoring pieces (≈3 × 1.8k chars max), then restore code order
            items = sorted(items, key=lambda t: -t[1])[:3]
            items.sort(key=lambda t: t[0].start_line)
            first = items[0][0]
            lines_out, last_end = [], 0
            for c, _, _ in items:
                ls = c.text.splitlines()
                skip = max(0, last_end - c.start_line + 1) if last_end else 0   # drop overlap
                if last_end and c.start_line > last_end + 1:
                    lines_out.append("    # ... (lines omitted) ...")
                lines_out += ls[skip:]
                last_end = max(last_end, c.end_line)
            total = len(items)
            combo = Chunk(repo=first.repo, path=first.path, kind=first.kind,
                          name=f"{key[3]} ({total} relevant parts)", text="\n".join(lines_out),
                          start_line=first.start_line, end_line=last_end, meta=first.meta)
            best = max(items, key=lambda t: t[1])
            merged.append((combo, best[1], best[2]))
        return merged

    def ask(self, question: str, k: int = 8, path_filter: str | None = None,
            chat: bool = False, max_tokens: int = 1500) -> dict:
        """Retrieve relevant excerpts and (if an API key is set) generate a cited answer.
        chat=True keeps the last few turns so follow-ups like "and the second one?" work."""
        query = question
        if chat and self.history:
            # enrich the search with the previous question for pronoun-style follow-ups
            query = self.history[-2]["content"].split("QUESTION:")[-1] + "\n" + question
        hits = self.retrieve(query, k=k, path_filter=path_filter)
        context = format_context(hits)
        sources = [{"n": n, "cite": c.cite, "kind": c.kind, "name": c.name, "score": round(s, 4)}
                   for n, (c, s, _) in enumerate(hits, 1)]

        if self._client is None:
            answer = ("(No GEMINI_API_KEY set — showing retrieved excerpts only.)\n\n" + context)
            return {"answer": answer, "sources": sources, "hits": hits}

        from google.genai import types

        user = (f"REPOSITORY MAP (every indexed file and what it defines):\n{self.index.repo_map}\n\n"
                f"RETRIEVED EXCERPTS:\n{context}\n\nQUESTION: {question}")
        history = self.history[-6:] if chat else []
        contents = [types.Content(role=("model" if m["role"] == "assistant" else "user"),
                                   parts=[types.Part(text=m["content"])]) for m in history]
        contents.append(types.Content(role="user", parts=[types.Part(text=user)]))
        resp = self._client.models.generate_content(
            model=self.llm_model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=max_tokens))
        answer = resp.text or ""
        if chat:
            # store a compact user turn (no bulky context) to keep follow-ups cheap
            self.history += [{"role": "user", "content": f"QUESTION: {question}"},
                             {"role": "assistant", "content": answer}]
        return {"answer": answer, "sources": sources, "hits": hits}

    def reset_chat(self):
        self.history = []

    # ------------------------------------------------------------ pretty print
    def print_answer(self, result: dict):
        print(result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['cite']}  — {s['kind']}: {s['name']}")

    def explain(self, question: str, k: int = 8):
        """Debug view: what retrieval found and why (BM25 / dense scores)."""
        for n, (c, s, parts) in enumerate(self.retrieve(question, k=k), 1):
            d = f" dense={parts['dense']:.3f}" if parts["dense"] is not None else ""
            print(f"[{n}] fused={s:.4f} bm25={parts['bm25']:.2f}{d}  {c.cite}  ({c.kind}: {c.name})")
            print(textwrap.indent(textwrap.shorten(c.text, 220), "      "))
