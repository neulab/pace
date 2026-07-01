#!/usr/bin/env python3
"""Watch live progress of all running BEIR reranking jobs.

Usage (in a separate terminal):
    python3 watch_progress.py
"""

import glob
import json
import os
import time

from tqdm import tqdm

STATUS_DIR = os.path.join(os.path.dirname(__file__), "beir_results", "_status")


def short_name(model: str) -> str:
    parts = model.split("/")
    # Keep last 2 segments, e.g. "azure_ai/gpt-5.2" or "minimax-m2p5"
    return "/".join(parts[-2:]) if len(parts) >= 2 else model


def fmt_elapsed(secs: float) -> str:
    s = int(secs)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_statuses():
    statuses = {}
    for path in glob.glob(os.path.join(STATUS_DIR, "*.json")):
        try:
            with open(path) as f:
                s = json.load(f)
            statuses[s["model"]] = s
        except Exception:
            pass
    return statuses


def main():
    bars: dict[str, tqdm] = {}        # model -> tqdm bar
    start_times: dict[str, float] = {}  # model -> first seen time
    positions: dict[str, int] = {}

    print(f"Watching {STATUS_DIR}  (Ctrl+C to quit)\n")

    try:
        while True:
            statuses = load_statuses()
            now = time.time()

            for model, s in sorted(statuses.items()):
                status = s.get("status", "running")
                stage = s.get("stage", "?")
                completed = s.get("completed", 0)
                total = s.get("total", 0)
                started_at = s.get("started_at", now)

                if model not in bars:
                    pos = len(bars)
                    positions[model] = pos
                    start_times[model] = started_at
                    desc = f"{short_name(model):<38}"
                    bars[model] = tqdm(
                        total=max(total, 1),
                        desc=desc,
                        position=pos,
                        leave=True,
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar:25}{r_bar}",
                    )

                bar = bars[model]

                # Update total if we now know it
                if total > 0 and bar.total != total:
                    bar.total = total
                    bar.refresh()

                # Advance completed count (tqdm only goes forward)
                delta = max(0, completed - bar.n)
                if delta > 0:
                    bar.update(delta)

                # Postfix: stage + elapsed
                elapsed = fmt_elapsed(now - start_times[model])
                if status == "done":
                    ndcg = s.get("ndcg10")
                    postfix = f"DONE  [{elapsed}]"
                    if ndcg is not None:
                        postfix += f"  NDCG@10={ndcg:.4f}"
                    bar.set_postfix_str(postfix)
                    bar.n = bar.total
                    bar.refresh()
                elif status == "skipped":
                    bar.set_postfix_str(f"SKIPPED")
                    bar.n = bar.total
                    bar.refresh()
                elif status == "failed":
                    bar.set_postfix_str(f"FAILED [{elapsed}]")
                    bar.refresh()
                else:
                    bar.set_postfix_str(f"{stage}  [{elapsed}]")
                    bar.refresh()

            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        for bar in bars.values():
            bar.close()
        print()


if __name__ == "__main__":
    main()
