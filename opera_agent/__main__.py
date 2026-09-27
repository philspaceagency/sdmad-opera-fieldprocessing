"""Command line.

  # talk to it (needs GEMINI_API_KEY; with OPENROUTER_API_KEY set it falls back to Qwen when Gemini is busy)
  python -m opera_agent "process the GoPro videos in /data/13MAY2025 with the GPX in /data/GPX"
  python -m opera_agent --chat --data-root /data
  python -m opera_agent --provider openrouter --chat --data-root /data      # Qwen only

  # or run the pipeline directly, no LLM involved
  python -m opera_agent inspect /data/13MAY2025/DCIM --gpx /data/13MAY2025/GPX/13MAY2025.GPX
  python -m opera_agent process /data/13MAY2025/DCIM /data/13MAY2025_processed --gpx /data/13MAY2025/GPX/13MAY2025.GPX \
      --yolo-model models/yolo11l-benthic-cls.pt          # optional: classify after geotagging
  python -m opera_agent classify /data/13MAY2025_processed --yolo-model models/yolo11l-benthic-cls.pt
"""
import argparse
import json
import os
import sys

from . import pipeline as P


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "classify":
        p = argparse.ArgumentParser(prog="opera_agent classify")
        p.add_argument("cmd"); p.add_argument("output_dir", help="a folder produced by process")
        p.add_argument("--yolo-model", default=os.environ.get("OPERA_YOLO_MODEL"), help="classification .pt")
        p.add_argument("--conf", type=float, default=0.0)
        p.add_argument("--organize", choices=["copy", "move", "none"], default="copy")
        a = p.parse_args()
        if not a.yolo_model:
            p.error("classify needs --yolo-model (or OPERA_YOLO_MODEL)")
        from .classify import classify_frames
        print(json.dumps(classify_frames(a.output_dir, a.yolo_model, conf=a.conf, organize=a.organize), indent=2))
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("inspect", "process"):
        p = argparse.ArgumentParser(prog="opera_agent")
        p.add_argument("cmd"); p.add_argument("videos_dir")
        p.add_argument("output_dir", nargs="?")
        p.add_argument("--gpx", action="append")
        p.add_argument("--interval", type=float, default=1.0)
        p.add_argument("--tz", default=P.LOCAL_TZ)
        p.add_argument("--clock-offset", type=float, default=0.0)
        p.add_argument("--max-gap", type=float, default=60.0)
        p.add_argument("--max-width", type=int)
        p.add_argument("--yolo-model", default=os.environ.get("OPERA_YOLO_MODEL"),
                       help="classify the geotagged frames with this YOLO .pt")
        p.add_argument("--conf", type=float, default=0.0, help="below this confidence: 'uncertain'")
        a = p.parse_args()
        if a.cmd == "inspect":
            print(json.dumps(P.inspect(a.videos_dir, a.gpx, a.tz, a.clock_offset), indent=2, default=str))
        else:
            if not a.output_dir:
                p.error("process needs an output_dir")
            rep = P.process_survey(a.videos_dir, a.output_dir, a.gpx, a.interval, a.tz, a.clock_offset,
                                   a.max_gap, a.max_width, model_path=a.yolo_model, classify_conf=a.conf)
            keys = ("geotagged_dir", "frames_total", "status_counts", "files", "classification")
            print(json.dumps({k: rep[k] for k in keys if k in rep}, indent=2, default=str))
        return

    p = argparse.ArgumentParser(prog="opera_agent")
    p.add_argument("request", nargs="?")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--data-root")
    p.add_argument("--provider", choices=["gemini", "openrouter"], default="gemini",
                   help="gemini (GEMINI_API_KEY) or openrouter (Qwen, OPENROUTER_API_KEY)")
    p.add_argument("--model", help="model id (default gemini-3.5-flash, or the Qwen model for openrouter); "
                                   "e.g. gemini-3.5-flash-lite, gemini-3.1-pro")
    p.add_argument("--gemini-fallback", action="append",
                   help="Gemini model(s) to try before Qwen (default gemini-3.5-flash-lite; 'none' to skip)")
    p.add_argument("--fallback", choices=["auto", "openrouter", "none"], default="auto",
                   help="when Gemini stays busy, continue on Qwen via OpenRouter "
                        "(auto: if OPENROUTER_API_KEY is set)")
    p.add_argument("--fallback-model", help="OpenRouter model for the fallback (default qwen/qwen3-235b-a22b-2507)")
    p.add_argument("--yolo-model", help="path to yolo-benthic-cls.pt (for classification)")
    a = p.parse_args()
    from .agent import OperaAgent
    agent = OperaAgent(data_root=a.data_root, llm_model=a.model, model_path=a.yolo_model, provider=a.provider,
                       fallback=None if a.fallback == "none" else a.fallback, fallback_model=a.fallback_model,
                       gemini_fallbacks=None if not a.gemini_fallback else
                       [m for m in a.gemini_fallback if m != "none"])
    if a.chat or not a.request:
        agent.chat()
    else:
        print(agent.run(a.request))


if __name__ == "__main__":
    main()
