"""Command line.

  # API keys: environment variables, or a key file (see opera_agent/keys.py): --keys-file PATH,
  # $OPERA_KEYS_FILE, ./api_keys.txt or ~/.config/opera/api_keys.txt

  # talk to it (needs GEMINI_API_KEY; with OPENROUTER_API_KEY set it falls back to Qwen when Gemini is busy)
  python -m opera_agent "process the GoPro videos in /data/13MAY2025 with the GPX in /data/GPX"
  python -m opera_agent --chat --data-root /data
  python -m opera_agent --provider openrouter --chat --data-root /data      # Qwen only

  # or run the pipeline directly, no LLM involved
  python -m opera_agent inspect /data/13MAY2025/DCIM --gpx /data/13MAY2025/GPX/13MAY2025.GPX
  python -m opera_agent process /data/13MAY2025/DCIM /data/13MAY2025_processed --gpx /data/13MAY2025/GPX/13MAY2025.GPX \
      --yolo-model models/yolo11l-benthic-cls.pt          # optional: classify after geotagging
  python -m opera_agent classify /data/13MAY2025_processed --yolo-model models/yolo11l-benthic-cls.pt
  python -m opera_agent classify-images /data/frames /data/frames_classified --yolo-model models/yolo11l-benthic-cls.pt
"""
import argparse
import json
import os
import sys

from . import pipeline as P


def main():
    # --keys-file works with every subcommand; settings found there (API keys, OPERA_YOLO_MODEL, ...) go into
    # the environment before the options below read their defaults from it
    from .keys import load_keys, pop_keys_file_arg
    load_keys(pop_keys_file_arg(sys.argv))

    if len(sys.argv) > 1 and sys.argv[1] in ("classify", "classify-images"):
        images = sys.argv[1] == "classify-images"
        p = argparse.ArgumentParser(prog=f"opera_agent {sys.argv[1]}")
        p.add_argument("cmd")
        if images:
            p.add_argument("input_dir", help="folder of images (sub-folders are kept)")
            p.add_argument("output_dir")
        else:
            p.add_argument("output_dir", help="a folder produced by process")
            p.add_argument("--organize", choices=["copy", "move", "none"], default="copy")
        p.add_argument("--yolo-model", default=os.environ.get("OPERA_YOLO_MODEL"), help="classification .pt")
        p.add_argument("--conf", type=float, default=0.0)
        p.add_argument("--no-annotate", action="store_true", help="don't draw the class probabilities on the images")
        p.add_argument("--tta", choices=["none", "hflip", "flips"], default="hflip",
                       help="average over flipped views (default hflip)")
        if not images:
            p.add_argument("--smooth", type=float, default=0.0, metavar="SECONDS",
                           help="average probabilities over neighbouring frames, e.g. 3 (default off)")
        a = p.parse_args()
        if not a.yolo_model:
            p.error(f"{a.cmd} needs --yolo-model (or OPERA_YOLO_MODEL)")
        from .classify import classify_frames, classify_images
        if images:
            res = classify_images(a.input_dir, a.output_dir, a.yolo_model, conf=a.conf, annotate=not a.no_annotate,
                                  tta=a.tta)
        else:
            res = classify_frames(a.output_dir, a.yolo_model, conf=a.conf, organize=a.organize,
                                  annotate=not a.no_annotate, tta=a.tta, smooth_s=a.smooth)
        print(json.dumps(res, indent=2))
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
        p.add_argument("--tta", choices=["none", "hflip", "flips"], default="hflip")
        p.add_argument("--smooth", type=float, default=0.0, metavar="SECONDS",
                       help="classification: average probabilities over neighbouring frames (default off)")
        a = p.parse_args()
        if a.cmd == "inspect":
            print(json.dumps(P.inspect(a.videos_dir, a.gpx, a.tz, a.clock_offset), indent=2, default=str))
        else:
            if not a.output_dir:
                p.error("process needs an output_dir")
            rep = P.process_survey(a.videos_dir, a.output_dir, a.gpx, a.interval, a.tz, a.clock_offset,
                                   a.max_gap, a.max_width, model_path=a.yolo_model, classify_conf=a.conf,
                                   classify_tta=a.tta, classify_smooth_s=a.smooth)
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
