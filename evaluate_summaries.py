import argparse
import json
import re
from typing import Dict, List, Set, Tuple


Triple = Tuple[str, str, str]


def canonical_text(value: str) -> str:
    """Normalize text so summary triples and gold edges can be compared robustly."""
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.strip("<>")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.lower()


def parse_summary_triple(triple_str: str) -> Triple:
    """Parse '<subj> <pred> "obj" .' format into a normalized triple."""
    # Matches objects with escaped quotes and trailing dot.
    match = re.match(r'^\s*(<[^>]+>)\s+(<[^>]+>)\s+"((?:[^"\\]|\\.)*)"\s*\.\s*$', triple_str)
    if not match:
        raise ValueError(f"Unsupported summary triple format: {triple_str}")
    subj, pred, obj = match.groups()
    return (canonical_text(subj), canonical_text(pred), canonical_text(obj))


def normalize_gold_edge(edge: List[str]) -> Triple:
    """Normalize [subject, predicate, object] edge from evaluation subset."""
    if len(edge) != 3:
        raise ValueError(f"Gold edge must have 3 elements, got: {edge}")
    subj, pred, obj = edge
    return (canonical_text(subj), canonical_text(pred), canonical_text(obj))


def compute_metrics(pred: Set[Triple], gold: Set[Triple]) -> Dict[str, float]:
    """Compute TP/FP/FN and precision/recall/F1."""
    tp = len(pred.intersection(gold))
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_evaluation_subset(path: str) -> Dict[int, Dict]:
    """Load evaluation records keyed by record index."""
    records: Dict[int, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            records[idx] = json.loads(line)
    return records


def load_outputs(path: str) -> List[Dict]:
    """Load generated output records from JSONL."""
    outputs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outputs.append(json.loads(line))
    return outputs


def evaluate(outputs_path: str, evaluation_path: str) -> Dict:
    """Evaluate generated summaries against gold edges."""
    eval_records = load_evaluation_subset(evaluation_path)
    outputs = load_outputs(outputs_path)

    per_record = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    evaluated = 0

    for out in outputs:
        # Some generators (e.g., IRES outputs) do not include a status field.
        # Only skip rows when status is explicitly present and not "ok".
        status = out.get("status")
        if status is not None and status != "ok":
            continue

        rec_idx = out.get("output", {}).get("record_index")
        if rec_idx is None or rec_idx not in eval_records:
            continue

        gold_edges = eval_records[rec_idx].get("edges", [])
        pred_triples = out.get("output", {}).get("summary", [])

        pred_set = {parse_summary_triple(t) for t in pred_triples}
        gold_set = {normalize_gold_edge(e) for e in gold_edges}

        metrics = compute_metrics(pred_set, gold_set)
        per_record.append(
            {
                "record_index": rec_idx,
                "question": eval_records[rec_idx].get("question"),
                "num_pred": len(pred_set),
                "num_gold": len(gold_set),
                **metrics,
            }
        )

        micro_tp += int(metrics["tp"])
        micro_fp += int(metrics["fp"])
        micro_fn += int(metrics["fn"])
        macro_p += metrics["precision"]
        macro_r += metrics["recall"]
        macro_f1 += metrics["f1"]
        evaluated += 1

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = (macro_p / evaluated) if evaluated > 0 else 0.0
    macro_recall = (macro_r / evaluated) if evaluated > 0 else 0.0
    macro_f1 = (macro_f1 / evaluated) if evaluated > 0 else 0.0

    return {
        "evaluated_records": evaluated,
        "micro": {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "per_record": per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries vs evaluation subset edges")
    parser.add_argument("--outputs-jsonl", required=True, help="Path to generated outputs JSONL")
    parser.add_argument("--evaluation-jsonl", required=True, help="Path to evaluation_subset JSONL")
    parser.add_argument("--report-json", default=None, help="Optional path to save full evaluation report JSON")
    args = parser.parse_args()

    report = evaluate(args.outputs_jsonl, args.evaluation_jsonl)

    print("=== Evaluation Summary ===")
    print(f"Evaluated records: {report['evaluated_records']}")
    print(
        "Micro P/R/F1: "
        f"{report['micro']['precision']:.4f} / "
        f"{report['micro']['recall']:.4f} / "
        f"{report['micro']['f1']:.4f}"
    )
    print(
        "Macro P/R/F1: "
        f"{report['macro']['precision']:.4f} / "
        f"{report['macro']['recall']:.4f} / "
        f"{report['macro']['f1']:.4f}"
    )

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report to: {args.report_json}")


if __name__ == "__main__":
    main()
