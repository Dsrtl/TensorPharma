"""Integrity checks for the committed result artifacts.

Standard library only, no model artifacts or datasets needed:

    python3 tests/test_results_integrity.py

Run from the TensorPharma folder. Exit code 0 means every check passed.

These guard the claims made in the README and on the demo site. One of them
exists because two feature-importance exports were once committed with
truncation warnings written into them as data rows.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)


def load_json(relative: str):
    path = ROOT / relative
    check(path.exists(), f"missing file: {relative}")
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def test_selected_model_is_ridge() -> None:
    """The README and the demo both claim Ridge was selected on validation."""
    summary = load_json("results/rank_benchmark_cancer_only/summary.json")
    if summary is None:
        return

    check(
        summary.get("selected_model") == "ridge_1000",
        f"selected_model is {summary.get('selected_model')!r}, expected 'ridge_1000'",
    )
    check(
        summary.get("benchmark_complete") is True,
        "benchmark_complete is not True",
    )

    models = summary.get("models", {})
    check(len(models) == 6, f"expected 6 model families, found {len(models)}")

    # Selection must be by validation. The winner must hold the best
    # validation Spearman even though another family wins on test.
    best_validation = max(models.values(), key=lambda m: m["validation_spearman"])
    ridge = models.get("fingerprint_ridge")
    check(ridge is not None, "fingerprint_ridge family missing from summary")
    if ridge is not None:
        check(
            best_validation is ridge,
            "fingerprint_ridge does not hold the best validation Spearman",
        )

    best_test = max(models.values(), key=lambda m: m["test_spearman"])
    check(
        best_test is not ridge,
        "expected a different family to win on test; the "
        "validation-only selection rule is the point of this check",
    )


def test_target_expression_did_not_help() -> None:
    """A negative result. The incumbent must still be retained."""
    summary = load_json("results/target_expression_filtered_test/summary.json")
    if summary is None:
        return

    check(
        summary.get("beats_incumbent_on_validation") is False,
        "target-expression run claims it beat the incumbent",
    )
    check(
        summary.get("incumbent") == "ridge_1000",
        f"incumbent is {summary.get('incumbent')!r}, expected 'ridge_1000'",
    )

    models = summary.get("models", {})
    control = models.get("lightgbm_control", {}).get("validation_spearman")
    treated = models.get("lightgbm_target_expression", {}).get("validation_spearman")
    check(
        control is not None and treated is not None,
        "target-expression summary is missing one of the two model entries",
    )
    if control is not None and treated is not None:
        check(
            treated < control,
            f"target-expression validation Spearman {treated} is not below "
            f"the control {control}; the reported negative result would be wrong",
        )


def test_step3_has_one_surviving_association() -> None:
    """Exactly one of the eight pre-specified links survived correction."""
    path = ROOT / "results/step3_evidence/association_results.csv"
    check(path.exists(), "missing results/step3_evidence/association_results.csv")
    if not path.exists():
        return

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    check(len(rows) == 8, f"expected 8 pre-specified links, found {len(rows)}")

    survivors = [r for r in rows if r["fdr_association"].strip().lower() == "true"]
    check(
        len(survivors) == 1,
        f"expected exactly 1 surviving association, found {len(survivors)}",
    )
    if len(survivors) == 1:
        check(
            survivors[0]["link_id"] == "ERBB2_breast",
            f"survivor is {survivors[0]['link_id']!r}, expected 'ERBB2_breast'",
        )

    # Every tested row needs a q-value; untested rows must not claim one.
    for row in rows:
        tested = row["status"] == "tested_exploratory"
        has_q = bool(row["q_value"].strip())
        check(
            tested == has_q,
            f"{row['link_id']}: status {row['status']!r} disagrees with "
            f"q_value {row['q_value']!r}",
        )


def test_feature_importance_exports_are_clean() -> None:
    """Regression guard: these were once committed truncated."""
    for directory in ("rank_benchmark", "rank_benchmark_cancer_only"):
        relative = f"results/{directory}/lightgbm_feature_importance.csv"
        path = ROOT / relative
        check(path.exists(), f"missing file: {relative}")
        if not path.exists():
            continue

        text = path.read_text()
        check(
            "truncated" not in text and "Total output lines" not in text,
            f"{relative} contains truncation artifacts instead of data",
        )

        with path.open() as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)

        check(
            header == ["feature", "gain"],
            f"{relative} header is {header}, expected ['feature', 'gain']",
        )
        check(len(rows) > 1000, f"{relative} has only {len(rows)} data rows")

        malformed = [r for r in rows if len(r) != 2]
        check(not malformed, f"{relative} has {len(malformed)} malformed rows")

        for row in rows:
            if len(row) == 2:
                try:
                    float(row[1])
                except ValueError:
                    check(False, f"{relative}: non-numeric gain {row[1]!r}")
                    break


def test_examples_parse() -> None:
    for relative in (
        "examples/imatinib_chain.json",
        "examples/heldout_test_chain.json",
    ):
        payload = load_json(relative)
        check(
            isinstance(payload, (dict, list)),
            f"{relative} did not parse into an object or array",
        )


def test_no_cluster_account_in_docs() -> None:
    """The repository is public; runbooks must use placeholders."""
    for path in ROOT.rglob("*.md"):
        text = path.read_text(errors="ignore")
        relative = path.relative_to(ROOT)
        check(
            "ab153483" not in text,
            f"{relative} contains a cluster username; use <user> instead",
        )
        check(
            "newton.ist.ucf.edu" not in text,
            f"{relative} contains a cluster hostname; use <cluster-host> instead",
        )


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()

    if FAILURES:
        print(f"FAILED  {len(FAILURES)} of {CHECKS} checks\n")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print(f"ok  {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
