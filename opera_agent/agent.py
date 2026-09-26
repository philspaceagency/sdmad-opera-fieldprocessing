"""Natural-language agent: "process the files in X" → geotagged GoPro frames.

Gemini plans and calls the pipeline tools; the fieldprocessing repository is retrieved
(RAG) when it needs to know how the original workflow does something."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .tools import TOOLS, ToolRunner

DEFAULT_LLM = "gemini-2.5-flash"
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
5. Classification and video concatenation are optional. Do them only when asked.
6. For questions about the original method, use search_workflow_docs and cite the file it came from.
Be concise. Give exact paths and numbers. Never invent paths or results."""


class OperaAgent:
    def __init__(self, data_root: str | None = None, llm_model: str = DEFAULT_LLM, api_key: str | None = None,
                 rag=None, docs_index: str | None = None, model_path: str | None = None,
                 verbose: bool = True, max_steps: int = 25):
        from google import genai
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.llm_model, self.data_root, self.verbose, self.max_steps = llm_model, data_root, verbose, max_steps
        if rag is None:
            rag = self._load_docs(docs_index)
        self.tools = ToolRunner(rag=rag, default_model_path=model_path, log=self._log)
        self.messages: list = []   # list of google.genai.types.Content

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
    def run(self, request: str) -> str:
        """Send a request; the agent calls tools until it has an answer. Keeps the
        conversation, so follow-ups ("yes, use UTC", "now classify them") work."""
        from google.genai import types

        system = SYSTEM + (f"\nData root (search here first): {self.data_root}" if self.data_root else "")
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=4000,
                                             tools=[types.Tool(function_declarations=TOOLS)])
        self.messages.append(types.Content(role="user", parts=[types.Part(text=request)]))
        for _ in range(self.max_steps):
            resp = self.client.models.generate_content(model=self.llm_model, contents=self.messages, config=config)
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None or not cand.content.parts:
                reason = getattr(cand, "finish_reason", None) if cand else getattr(resp, "prompt_feedback", None)
                return f"Gemini returned no content (finish reason: {reason})."
            content = cand.content
            self.messages.append(content)
            calls = [p.function_call for p in content.parts if p.function_call]
            if not calls:
                return "".join(p.text for p in content.parts if p.text)
            response_parts = []
            for c in calls:
                args = dict(c.args or {})
                self._log(f"→ {c.name}({json.dumps(args, ensure_ascii=False)[:200]})")
                out = self.tools(c.name, args)
                if "error" in out:
                    self._log(f"  ! {out['error']}")
                dumped = json.dumps(out, default=str)          # plain JSON types only (no Path/numpy)
                payload = json.loads(dumped) if len(dumped) <= 20000 else {"truncated": True, "preview": dumped[:20000]}
                # Part.from_function_response() has no `id` argument; build the Part directly
                response_parts.append(types.Part(function_response=types.FunctionResponse(
                    name=c.name, response={"result": payload}, id=getattr(c, "id", None))))
            self.messages.append(types.Content(role="user", parts=response_parts))
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
