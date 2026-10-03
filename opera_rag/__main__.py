"""Command line:
  python -m opera_rag build [--repo URL_OR_PATH ...] [--index ./opera_index] [--no-embed]
  python -m opera_rag ask "How are frames geotagged?" [--index ./opera_index] [-k 8]
  python -m opera_rag chat [--index ./opera_index]
  python -m opera_rag search "exiftool path"          # retrieval only, no LLM
"""
import argparse
import sys

from .rag import OperaRAG, DEFAULT_REPOS, DEFAULT_EMBED, DEFAULT_LLM


def main():
    from opera_agent.keys import load_keys, pop_keys_file_arg      # --keys-file, or ./api_keys.txt etc.
    load_keys(pop_keys_file_arg(sys.argv))

    p = argparse.ArgumentParser(prog="opera_rag", description="RAG over OpERA repositories")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="index repositories")
    b.add_argument("--repo", action="append", help="git URL or local path (repeatable)")
    b.add_argument("--index", default="./opera_index")
    b.add_argument("--embed-model", default=DEFAULT_EMBED)
    b.add_argument("--no-embed", action="store_true", help="keyword search only")
    b.add_argument("--no-docs", action="store_true", help="skip PDF/PPTX")

    for name in ("ask", "search", "chat"):
        s = sub.add_parser(name)
        if name != "chat":
            s.add_argument("question")
        s.add_argument("--index", default="./opera_index")
        s.add_argument("-k", type=int, default=8)
        s.add_argument("--model", default=DEFAULT_LLM)
        s.add_argument("--only", help="restrict to paths containing this text, e.g. scripts/")

    a = p.parse_args()
    if a.cmd == "build":
        rag = OperaRAG.build(a.repo or DEFAULT_REPOS,
                             embed_model=None if a.no_embed else a.embed_model,
                             include_docs=not a.no_docs)
        rag.save(a.index)
        print(f"Saved to {a.index}")
        return

    rag = OperaRAG.load(a.index, llm_model=a.model)
    if a.cmd == "search":
        rag.explain(a.question, k=a.k)
    elif a.cmd == "ask":
        rag.print_answer(rag.ask(a.question, k=a.k, path_filter=a.only))
    else:
        print("Ask about the code (empty line to quit, 'reset' to clear history).")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            if q == "reset":
                rag.reset_chat(); continue
            rag.print_answer(rag.ask(q, k=a.k, path_filter=a.only, chat=True))


if __name__ == "__main__":
    main()
