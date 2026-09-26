"""Command line.

  # talk to it (needs GEMINI_API_KEY)
  python -m opera_agent "process the GoPro videos in /data/13MAY2025 with the GPX in /data/GPX"
  python -m opera_agent --chat --data-root /data

  # or run the pipeline directly, no LLM involved
  python -m opera_agent inspect /data/13MAY2025/DCIM --gpx /data/13MAY2025/GPX/13MAY2025.GPX
  python -m opera_agent process /data/13MAY2025/DCIM /data/13MAY2025_processed --gpx /data/13MAY2025/GPX/13MAY2025.GPX
"""
import argparse
import json
import sys

from . import pipeline as P


def main():
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
        a = p.parse_args()
        if a.cmd == "inspect":
            print(json.dumps(P.inspect(a.videos_dir, a.gpx, a.tz, a.clock_offset), indent=2, default=str))
        else:
            if not a.output_dir:
                p.error("process needs an output_dir")
            rep = P.process_survey(a.videos_dir, a.output_dir, a.gpx, a.interval, a.tz, a.clock_offset,
                                   a.max_gap, a.max_width)
            print(json.dumps({k: rep[k] for k in ("geotagged_dir", "frames_total", "status_counts", "files")}, indent=2))
        return

    p = argparse.ArgumentParser(prog="opera_agent")
    p.add_argument("request", nargs="?")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--data-root")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--yolo-model", help="path to yolo-benthic-cls.pt (for classification)")
    a = p.parse_args()
    from .agent import OperaAgent
    agent = OperaAgent(data_root=a.data_root, llm_model=a.model, model_path=a.yolo_model)
    if a.chat or not a.request:
        agent.chat()
    else:
        print(agent.run(a.request))


if __name__ == "__main__":
    main()
