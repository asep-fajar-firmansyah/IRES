#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Triple = Tuple[str, str, str]


def normalize_entity(value: str) -> str:
    return value.replace(" ", "_")


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_graph_stats(edges: List[List[str]]) -> Tuple[List[Triple], Counter, Dict[str, int], Dict[str, List[Triple]], Dict[str, List[Triple]]]:
    triples: List[Triple] = []
    relation_freq: Counter = Counter()
    neighbors: Dict[str, set] = defaultdict(set)
    outgoing: Dict[str, List[Triple]] = defaultdict(list)
    incoming: Dict[str, List[Triple]] = defaultdict(list)

    for s_raw, p_raw, o_raw in edges:
        s = normalize_entity(str(s_raw))
        p = str(p_raw)
        o = normalize_entity(str(o_raw))
        t = (s, p, o)
        triples.append(t)
        relation_freq[p] += 1
        neighbors[s].add(o)
        neighbors[o].add(s)
        outgoing[s].append(t)
        incoming[o].append(t)

    degree = {node: len(nb) for node, nb in neighbors.items()}
    return triples, relation_freq, degree, outgoing, incoming


def compute_frequency_features(
    relation_freq: Counter, degree: Dict[str, int], outgoing: Dict[str, List[Triple]]
) -> Dict[str, float]:
    # IRES frequency feature: d_i + freq(r) aggregated over outgoing relations of node i
    features: Dict[str, float] = {}
    nodes = set(degree.keys()) | set(outgoing.keys())
    for node in nodes:
        di = degree.get(node, 0)
        rel_sum = sum(relation_freq[p] for _, p, _ in outgoing.get(node, []))
        features[node] = float(di + rel_sum)
    return features


def rank_triples_for_entity(
    entity_norm: str,
    outgoing: Dict[str, List[Triple]],
    incoming: Dict[str, List[Triple]],
    features: Dict[str, float],
    relation_freq: Counter,
) -> List[Triple]:
    # Outgoing: entity is subject (s, p, o)
    # Incoming: entity is object, but we want to show entity as subject (o, p_inverse, s) - but keep as-is for scoring
    candidates = outgoing.get(entity_norm, []) + incoming.get(entity_norm, [])
    if not candidates:
        return []

    # Score with frequency feature + relation popularity + object feature
    scored = []
    for triple in candidates:
        s, p, o = triple
        score = features.get(s, 0.0) + relation_freq.get(p, 0) + features.get(o, 0.0)
        scored.append((score, triple))

    scored.sort(key=lambda item: (-item[0], item[1][1], item[1][2]))
    return [t for _, t in scored]


def format_summary_triple(triple: Triple) -> str:
    s, p, o = triple
    object_text = o.replace("_", " ")
    return f"<{s}> <{p}> \"{object_text}\" ."


def build_output_entry(record: Dict, record_index: int, k: int) -> Dict:
    question = str(record.get("question", ""))
    graph_id = int(record.get("graph_id", -1))
    edges = record.get("edges", []) or []
    topic_entities_raw = record.get("topic_entities", []) or []

    triples, relation_freq, degree, outgoing, incoming = build_graph_stats(edges)
    _ = triples  # triples are used indirectly through outgoing/relation stats
    features = compute_frequency_features(relation_freq, degree, outgoing)

    topic_entities_norm = [normalize_entity(str(e)) for e in topic_entities_raw]

    summary_lines: List[str] = []
    for entity in topic_entities_norm:
        ranked = rank_triples_for_entity(entity, outgoing, incoming, features, relation_freq)
        topk = ranked[:k]
        summary_lines.extend(format_summary_triple(t) for t in topk)

    return {
        "meta": {
            "mode": "dataset",
            "dataset_json": "datasets/M3GQA/all_single_setting/evaluation_subset.jsonl",
            "record_index": record_index,
            "graph_id": graph_id,
            "top_k": k,
        },
        "output": {
            "record_index": record_index,
            "question": question,
            "entities": [f"<{e}>" for e in topic_entities_norm],
            "summary": summary_lines,
        },
    }


def process_dataset(dataset_jsonl: Path, output_dir: Path, ks: List[int], limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(load_jsonl(dataset_jsonl))
    if limit > 0:
        rows = rows[:limit]

    for k in ks:
        out_path = output_dir / f"ires_m3gqa_top{k}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(rows):
                result = build_output_entry(row, idx, k)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} records -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRES summarization for M3GQA all_single_setting dataset.")
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path("/home/asepfajarfirmansyah/Documents/GitHub/DL4MES/datasets/M3GQA/all_single_setting/evaluation_subset.jsonl"),
        help="Path to evaluation_subset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory where top-k JSONL files are written",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 5],
        help="Top-k values to generate (e.g., --ks 1 2 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional record limit for testing. 0 means full dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.dataset_jsonl, args.output_dir, args.ks, args.limit)


if __name__ == "__main__":
    main()
