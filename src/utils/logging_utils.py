"""Tiny run logger.

Every pipeline run (data stage now, training stage later) writes a
timestamped JSON-lines event log plus one final summary JSON. This is the
mechanism behind "store every number used in this project" -- nothing is
meant to live only in a terminal scrollback.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, out_dir: str | Path, run_name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.log_path = self.out_dir / f"{run_name}.log.jsonl"
        self._f = open(self.log_path, "a", encoding="utf-8")

    def _now(self) -> str:
        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": self._now(), "event": event, **fields}
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()
        print(f"[{record['ts']}] {event}: "
              f"{ {k: v for k, v in fields.items() if k != 'event'} }")

    def write_summary(self, summary: dict) -> Path:
        summary_path = self.out_dir / f"{self.run_name}.summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary_path

    def close(self) -> None:
        self._f.close()
