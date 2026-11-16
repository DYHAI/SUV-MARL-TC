"""Audit the 500-storm, 0--60 m HYCOM current archive.

The archive combines the original 200-event list with the additional 300-event
list.  The two downloaders import different modules, so the expected tags are
generated in isolated subprocesses to avoid module-level metadata mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


DEFAULT_DEPTHS = (0, 10, 20, 30, 40, 50, 60)
LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "union_region")
NAS_ROOT = "/mnt/nas/multi_auv_typhoon_tracking/union_current_depths"


def tags_from(module: str) -> list[str]:
    code = (
        "import importlib; "
        f"T=importlib.import_module('{module}'); "
        "print('\\n'.join(T.tag_of(s) for s in T.build_storm_list()))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def audit(depths: tuple[int, ...]) -> dict[str, object]:
    expected = sorted(set(tags_from("typhoons_200") + tags_from("typhoons_500")))
    if len(expected) != 500:
        raise RuntimeError(f"expected 500 unique storm tags, found {len(expected)}")

    result: dict[str, object] = {
        "expected_events": len(expected),
        "roots": [LOCAL_ROOT, NAS_ROOT],
        "depths": {},
    }
    for depth in depths:
        available: dict[str, list[str]] = {}
        for tag in expected:
            name = f"hycom_uv_{depth}m.nc"
            sources = [
                root for root in (LOCAL_ROOT, NAS_ROOT)
                if os.path.isfile(os.path.join(root, tag, name))
                and os.path.getsize(os.path.join(root, tag, name)) > 0
            ]
            if sources:
                available[tag] = sources

        missing = [tag for tag in expected if tag not in available]
        counts = {
            root: sum(root in sources for sources in available.values())
            for root in (LOCAL_ROOT, NAS_ROOT)
        }
        result["depths"][str(depth)] = {
            "available": len(available),
            "expected": len(expected),
            "missing": missing,
            "sources": counts,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--json")
    parser.add_argument("--print-missing-depths", action="store_true")
    args = parser.parse_args()

    depths = tuple(int(item) for item in args.depths.split(",") if item.strip())
    report = audit(depths)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    missing_depths = []
    for depth in depths:
        item = report["depths"][str(depth)]
        available = item["available"]
        missing = item["missing"]
        print(
            f"{depth:>2} m: {available:3}/500 "
            f"(local {item['sources'][LOCAL_ROOT]:3}, NAS {item['sources'][NAS_ROOT]:3}), "
            f"missing {len(missing):3}"
        )
        if missing:
            missing_depths.append(depth)

    if args.print_missing_depths:
        print("\n".join(map(str, missing_depths)))

    return 1 if missing_depths else 0


if __name__ == "__main__":
    raise SystemExit(main())
