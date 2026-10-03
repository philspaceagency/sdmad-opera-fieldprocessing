"""Natural-language agent: "process the files in X" → geotagged GoPro frames.

Gemini (or Qwen via OpenRouter when Gemini is busy, see llm.py) plans and calls the pipeline tools;
the fieldprocessing repository is retrieved (RAG) when it needs to know how the original workflow does something."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .llm import DEFAULT_GEMINI as DEFAULT_LLM, GEMINI_FALLBACKS, ModelUnavailableError, TransientLLMError, make_backend
from .tools import TOOLS, ToolRunner

FIELD_REPO = "https://github.com/philspaceagency/sdmad-opera-fieldprocessing.git"

SYSTEM = """You operate PhilSA's OpERA field-processing pipeline for underwater benthic surveys:
GoPro video (chapters GHzzxxxx/GXzzxxxx: zz = chapter, xxxx = recording) → frames every N seconds,
named frame_YYYY-MM-DD_HH-MM-SS.jpg from the video CreateDate (camera clock, Asia/Manila by default)
→ position and depth interpolated from the echosounder GPX (UTC times) → GPS written into each JPG's EXIF.

How to work:
1. If the user didn't give exact paths, use list_folder on the data root to find folders with videos and .gpx files.
   If several surveys are possible and the request doesn't say which, ask.
2. Always run inspect_survey before process_survey.
   - If a recording has NO overlap with the GPX, do not process it. Report the video and GPX time ranges and the
     likely cause (camera clock set to UTC → tz="UTC"; clock drift → clock_offset_s), then ask the user.
   - Partial overlap is fine: frames outside the track are kept in untagged/ and reported.
3. Default output_dir: a sibling of the videos folder named <videos folder>_processed, unless the user gives one.
   Defaults: interval_s=1, max_gap_s=60, full resolution. Change them only when the user asks.
4. After processing, run verify_geotags, then report briefly: output folder, frames geotagged vs total, untagged
   frames and why (before/after the track, GPS gaps), and the files (geotagged/, frame_data.csv, frames.geojson,
   qa_track_map.png).
5. Classification: when a YOLO model is configured, process_survey classifies the geotagged frames right after
   geotagging (classify=false only if the user says not to). Report the class counts and classified/<class>/,
   qa_class_map.png. To (re)classify an existing output without re-extracting, use classify_frames.
   Video concatenation is optional: only when asked.
6. For questions about the original method, use search_workflow_docs and cite the file it came from.
Be concise. Give exact paths and numbers. Never invent paths or results."""


class OperaAgent:
    """provider: "gemini" (default) or "openrouter" (Qwen via OpenRouter, key in OPENROUTER_API_KEY).
    When the active model stays busy (rate limit / overload / network) after `retries`, or doesn't exist for
    this key (404), the same conversation continues on the next model in the chain:
      primary → gemini_fallbacks (default gemini-3.5-flash-lite) → Qwen on OpenRouter.
    fallback: "auto" = add Qwen if OPENROUTER_API_KEY is set, "openrouter" = always, None = no Qwen.
    Each new request tries the primary model first again.
    keys_file: API keys from a text file (see keys.py); without it ./api_keys.txt and
    ~/.config/opera/api_keys.txt are used when present. Environment variables win over the file."""

    def __init__(self, data_root: str | None = None, llm_model: str | None = None, api_key: str | None = None,
                 rag=None, docs_index: str | None = None, model_path: str | None = None,
                 verbose: bool = True, max_steps: int = 25, provider: str = "gemini",
                 fallback: str | None = "auto", fallback_model: str | None = None, retries: int = 2,
                 gemini_fallbacks: list[str] | None = None, backend=None, fallback_backends: list | None = None,
                 keys_file: str | None = None):
        from .keys import load_keys
        load_keys(keys_file, log=print if verbose else (lambda *_: None))
        self.data_root, self.verbose, self.max_steps, self.retries = data_root, verbose, max_steps, retries
        self.backend = backend or make_backend(provider, llm_model, api_key)
        if fallback_backends is None:
            fallback_backends = []
            if provider == "gemini":
                for m in GEMINI_FALLBACKS if gemini_fallbacks is None else gemini_fallbacks:
                    if m != self.backend.model:
                        fallback_backends.append(make_backend("gemini", m, api_key))
                if fallback and (fallback == "openrouter" or os.environ.get("OPENROUTER_API_KEY")):
                    fallback_backends.append(make_backend("openrouter", fallback_model))
        self.fallbacks = list(fallback_backends)
        self._sleep = time.sleep
        model_path = model_path or os.environ.get("OPERA_YOLO_MODEL")
        if rag is None:
            rag = self._load_docs(docs_index)
        self.tools = ToolRunner(rag=rag, default_model_path=model_path, log=self._log)
        self.messages: list[dict] = []   # provider-neutral history (see llm.py)
        self._log(f"LLM: {self.backend.name}"
                  + (f" (fallback: {' → '.join(b.name for b in self.fallbacks)})" if self.fallbacks else ""))

    # -------------------------------------------------------------- docs (RAG)
    @staticmethod
    def _load_docs(docs_index):
        try:
            from opera_rag import OperaRAG
            if docs_index and Path(docs_index, "chunks.jsonl").exists():
                return OperaRAG.load(docs_index)
            rag = OperaRAG.build([FIELD_REPO], embed_model=None)
            if docs_index:
                rag.save(docs_index)
            return rag
        except Exception as e:
            print(f"(workflow docs search unavailable: {e})")
            return None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------- main loop
    def _complete(self, system: str) -> dict:
        """Ask the active model; on a transient error retry with backoff, then move to the fallback."""
        chain = [self.backend] + self.fallbacks
        err = None
        for i in range(self._active, len(chain)):
            b = chain[i]
            for attempt in range(self.retries + 1):
                try:
                    return b.complete(system, self.messages, TOOLS)
                except ModelUnavailableError as e:
                    err = e
                    self._log(f"  {b.name} is not available for this API key ({str(e)[:160]})")
                    break
                except TransientLLMError as e:
                    err = e
                    if attempt < self.retries:
                        wait = 2 ** (attempt + 1)
                        self._log(f"  {b.name} unavailable ({str(e)[:120]}); retrying in {wait}s")
                        self._sleep(wait)
            if i + 1 < len(chain):
                self._log(f"  {b.name} unavailable → switching to {chain[i + 1].name}")
                self._active = i + 1
        raise err

    def run(self, request: str) -> str:
        """Send a request; the agent calls tools until it has an answer. Keeps the
        conversation, so follow-ups ("yes, use UTC", "now classify them") work."""
        system = SYSTEM + (f"\nData root (search here first): {self.data_root}" if self.data_root else "")
        system += ("\nYOLO model configured: classification runs after geotagging." if self.tools.default_model_path
                   else "\nNo YOLO model configured: classification is unavailable unless the user gives model_path.")
        self.messages.append({"role": "user", "content": request})
        self._active = 0                                   # every request tries the primary model first
        for _ in range(self.max_steps):
            try:
                res = self._complete(system)
            except TransientLLMError as e:
                return f"No model available right now ({e}). Try again in a few minutes."
            if res["message"] is None:
                return res["text"]
            self.messages.append(res["message"])
            if not res["tool_calls"]:
                return res["text"]
            for c in res["tool_calls"]:
                self._log(f"→ {c['name']}({json.dumps(c['args'], ensure_ascii=False)[:200]})")
                out = self.tools(c["name"], c["args"])
                if "error" in out:
                    self._log(f"  ! {out['error']}")
                dumped = json.dumps(out, default=str)          # plain JSON types only (no Path/numpy)
                if len(dumped) > 20000:
                    dumped = json.dumps({"truncated": True, "preview": dumped[:20000]})
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                      "content": dumped, "_gemini_id": c.get("gemini_id")})
        return "Stopped: too many steps without finishing."

    def reset(self):
        self.messages = []

    def chat(self):
        print("Tell me what to process (empty line to quit).")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            print("\n" + self.run(q))
