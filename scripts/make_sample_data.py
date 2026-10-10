"""Create the committed sample dataset: 20 cells, every cell on a perfect 5-minute grid, with coordinates.

    python scripts/make_sample_data.py
    python scripts/make_sample_data.py --start 2026-02-24 --end 2026-02-27 --sites 20
Writes data/sample/sample_network_traffic.csv (+ .meta.json) and refuses to write if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, resolve_path  # noqa: E402
from src.data.generator import SYNTHETIC_WARNING, save_dataset  # noqa: E402
from src.data.sample import build_sample, validate_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the clean sample dataset (SIMULATED).")
    ap.add_argument("--sites", type=int)
    ap.add_argument("--start")
    ap.add_argument("--end", help="exclusive end date")
    a = ap.parse_args()
    cfg = get_config()
    s = cfg["sample"]
    if a.sites: s["n_sites"] = a.sites
    if a.start: s["start"] = a.start
    if a.end: s["end"] = a.end
    df = build_sample(cfg)
    checks = validate_sample(df, cfg)
    print(f"\n{SYNTHETIC_WARNING}\n")
    for name, (ok, detail) in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name:34s} {detail}")
    if not all(ok for ok, _ in checks.values()):
        sys.exit("Sample checks failed; nothing written.")
    out = resolve_path(cfg, "sample_dir") / s["filename"]
    save_dataset(df, out, cfg)
    print(f"\nWrote {out}  ({len(df):,} rows, {df['site_id'].nunique()} cells, "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}, {out.stat().st_size / 1e6:.1f} MB)")
    print("\nFirst cell, first 3 rows:")
    print(df[df.site_id == df.site_id.iloc[0]][["timestamp", "site_id", "region", "latitude", "longitude", "location",
                                                  "bandwidth_utilization_pct", "latency_ms"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
