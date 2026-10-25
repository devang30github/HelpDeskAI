"""Compare two runs.   python -m scripts.compare_runs <baseline_run> <candidate_run>
A run can be a folder path or a run id under reports/runs or reports/baselines."""
import argparse
import sys
from pathlib import Path

from app.core.config import BASE_DIR
from app.evaluation import compare as cmp


def _resolve(arg: str) -> Path | None:
    for candidate in (Path(arg), BASE_DIR / "reports" / "runs" / arg, BASE_DIR / "reports" / "baselines" / arg):
        if (candidate / "config.json").exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--force", action="store_true", help="compare even if the runs are not comparable")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="output markdown path")
    args = parser.parse_args()

    dir_a, dir_b = _resolve(args.baseline), _resolve(args.candidate)
    if dir_a is None or dir_b is None:
        print(f"Could not find run folder for: {args.baseline if dir_a is None else args.candidate}")
        return 1
    a, b = cmp.load_run(dir_a), cmp.load_run(dir_b)
    result = cmp.compare(a, b, args.bootstrap, args.seed)
    if result["problems"] and not args.force:
        print("These runs are not comparable:")
        for p in result["problems"]:
            print(f"  - {p}")
        print("Use --force to compare anyway (the report will say so).")
        return 1

    text = cmp.render(result, a, b)
    out = Path(args.out) if args.out else BASE_DIR / "reports" / "comparisons" / f"{dir_a.name}__vs__{dir_b.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())