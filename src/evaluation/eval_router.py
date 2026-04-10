"""
src/evaluation/eval_router.py
Router accuracy evaluation — intent classification Precision / Recall / F1

Test set: ACTION_ONLY 30 + QA_ONLY 30 + BOTH 30 = 90 total
Target  : ≥ 99% accuracy across all three classes

Usage (from project root):
    python -m src.evaluation.eval_router
    python -m src.evaluation.eval_router --test-cases data/eval/router_test_cases.json
    python -m src.evaluation.eval_router --out reports/router_eval.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from sklearn.metrics import classification_report, accuracy_score

from src.logging_config import setup_logging
from src.graph.router import router_node

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default test-cases path
# ---------------------------------------------------------------------------
_DEFAULT_TEST_CASES = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "eval" / "router_test_cases.json"
)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def eval_router(
    test_cases: list[dict],
    *,
    verbose: bool = True,
) -> dict:
    """
    Run every test case through router_node and compute classification metrics.

    Parameters
    ----------
    test_cases : list[dict]
        Each element must have:
            "input"  : str   — email text fed to the router
            "label"  : str   — ground-truth intent ("ACTION_ONLY" | "QA_ONLY" | "BOTH")
    verbose : bool
        If True, print per-sample progress and the final report to stdout.

    Returns
    -------
    dict with keys:
        "accuracy"  : float
        "report"    : dict  (sklearn classification_report output_dict=True)
        "errors"    : list  (samples where router returned None)
        "latency_s" : dict  {"mean": float, "total": float}
    """
    labels_true: list[str] = []
    labels_pred: list[str] = []
    errors: list[dict] = []
    latencies: list[float] = []

    total = len(test_cases)
    logger.info("Starting router evaluation on %d samples …", total)

    for i, case in enumerate(test_cases, start=1):
        user_input: str = case["input"]
        true_label: str = case["label"]

        # Build a minimal AgentState (only fields router_node reads are required)
        state = {
            "user_input": user_input,
            "error_messages": [],
        }

        t0 = time.perf_counter()
        result = router_node(state)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        predicted = result.get("intent")

        if predicted is None:
            logger.warning("[%d/%d] router returned None for: %.80s", i, total, user_input)
            errors.append({"index": i, "input": user_input, "true_label": true_label})
            # Treat as wrong prediction — use a sentinel so metrics reflect the failure
            predicted = "__ERROR__"

        labels_true.append(true_label)
        labels_pred.append(predicted)

        status = "✓" if predicted == true_label else "✗"
        if verbose:
            print(
                f"[{i:>3}/{total}] {status}  true={true_label:<12}  pred={predicted:<12}"
                f"  ({elapsed:.2f}s)  {user_input[:60]!r}"
            )

    # Filter out __ERROR__ entries for sklearn (it can't handle unknown labels cleanly)
    clean_true = [t for t, p in zip(labels_true, labels_pred) if p != "__ERROR__"]
    clean_pred = [p for t, p in zip(labels_true, labels_pred) if p != "__ERROR__"]

    target_names = ["ACTION_ONLY", "QA_ONLY", "BOTH"]
    report_dict: dict = classification_report(
        clean_true,
        clean_pred,
        labels=target_names,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy: float = accuracy_score(clean_true, clean_pred)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_latency = sum(latencies)

    # ── Print summary ────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 70)
        print(classification_report(
            clean_true,
            clean_pred,
            labels=target_names,
            target_names=target_names,
            zero_division=0,
        ))
        print(f"  Overall accuracy : {accuracy * 100:.2f}%  (target ≥ 99%)")
        print(f"  Errors (None)    : {len(errors)}")
        print(f"  Latency (mean)   : {mean_latency:.2f}s")
        print(f"  Latency (total)  : {total_latency:.1f}s")
        print("=" * 70)

        if accuracy >= 0.99:
            print("🎉  PASS — accuracy target met!")
        else:
            gap = (0.99 - accuracy) * total
            print(f"⚠️   FAIL — need {gap:.1f} more correct predictions to reach 99%.")

    return {
        "accuracy": accuracy,
        "report": report_dict,
        "errors": errors,
        "latency_s": {"mean": mean_latency, "total": total_latency},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the router node on a labelled test set."
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=_DEFAULT_TEST_CASES,
        help=f"Path to JSON test cases file (default: {_DEFAULT_TEST_CASES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample progress output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Load test cases ──────────────────────────────────────────────────────
    test_cases_path: Path = args.test_cases
    if not test_cases_path.exists():
        logger.error(
            "Test cases file not found: %s\n"
            "  Generate it with:  python -m src.evaluation.generate_router_testset",
            test_cases_path,
        )
        sys.exit(1)

    with open(test_cases_path, encoding="utf-8") as f:
        test_cases: list[dict] = json.load(f)
    logger.info("Loaded %d test cases from %s", len(test_cases), test_cases_path)

    # ── Run evaluation ───────────────────────────────────────────────────────
    results = eval_router(test_cases, verbose=not args.quiet)

    # ── Optionally save report ───────────────────────────────────────────────
    if args.out:
        out_path: Path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", out_path)

    # ── Exit code reflects pass/fail ─────────────────────────────────────────
    sys.exit(0 if results["accuracy"] >= 0.99 else 1)
