#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import FastRGCNConv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "torch_geometric is required for m3gqa_ires_train.py. "
        "Install dependencies first (e.g., `poetry install` in this repo, "
        "or install torch-geometric matching your torch version)."
    ) from exc


Triple = Tuple[str, str, str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_entity(text: str) -> str:
    return text.replace(" ", "_")


def denormalize_entity(text: str) -> str:
    return text.replace("_", " ")


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_NT_PATTERN = re.compile(r'^\s*<([^>]+)>\s+<([^>]+)>\s+(?:(<([^>]+)>)|("((?:[^"\\]|\\.)*)"))\s*\.\s*$')


def _parse_nt_line(line: str) -> Optional[Tuple[str, str, str]]:
    m = _NT_PATTERN.match(line)
    if not m:
        return None
    s = m.group(1)
    p = m.group(2)
    o_uri = m.group(4)
    o_lit = m.group(6)
    o = o_uri if o_uri is not None else (o_lit if o_lit is not None else "")
    return s, p, o


def _local_name(value: str) -> str:
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    if "#" in value:
        value = value.rsplit("#", 1)[-1]
    return value


def load_graph_nt_edges(graph_file: Path) -> List[List[str]]:
    edges: List[List[str]] = []
    if not graph_file.exists():
        return edges
    with graph_file.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_nt_line(line.strip())
            if parsed is None:
                continue
            s, p, o = parsed
            edges.append([_local_name(s), _local_name(p), _local_name(o)])
    return edges


class M3GQAGraphBuilder:
    def __init__(self) -> None:
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: List[str] = []
        self.relation_to_idx: Dict[str, int] = {}

        self.edge_src: List[int] = []
        self.edge_dst: List[int] = []
        self.edge_type: List[int] = []

        self.outgoing_by_record: Dict[int, Dict[str, List[Triple]]] = {}
        self.topic_entities_by_record: Dict[int, List[str]] = {}
        self.question_by_record: Dict[int, str] = {}
        self.graph_id_by_record: Dict[int, int] = {}

    def _entity_id(self, entity: str) -> int:
        if entity not in self.entity_to_idx:
            self.entity_to_idx[entity] = len(self.idx_to_entity)
            self.idx_to_entity.append(entity)
        return self.entity_to_idx[entity]

    def _relation_id(self, rel: str) -> int:
        if rel not in self.relation_to_idx:
            self.relation_to_idx[rel] = len(self.relation_to_idx)
        return self.relation_to_idx[rel]

    def add_record(self, record_index: int, row: Dict, edges_override: Optional[List[List[str]]] = None) -> None:
        question = str(row.get("question", ""))
        graph_id = int(row.get("graph_id", -1))
        topic_entities = [normalize_entity(str(e)) for e in row.get("topic_entities", [])]
        edges = edges_override if edges_override is not None else (row.get("edges", []) or [])

        self.question_by_record[record_index] = question
        self.graph_id_by_record[record_index] = graph_id
        self.topic_entities_by_record[record_index] = topic_entities

        outgoing: Dict[str, List[Triple]] = defaultdict(list)
        for s_raw, p_raw, o_raw in edges:
            s = normalize_entity(str(s_raw))
            p = str(p_raw)
            o = normalize_entity(str(o_raw))
            outgoing[s].append((s, p, o))

            s_idx = self._entity_id(s)
            o_idx = self._entity_id(o)
            r_idx = self._relation_id(p)

            self.edge_src.append(s_idx)
            self.edge_dst.append(o_idx)
            self.edge_type.append(r_idx)

        self.outgoing_by_record[record_index] = outgoing

    def build_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.tensor([self.edge_src, self.edge_dst], dtype=torch.long, device=device)
        edge_type = torch.tensor(self.edge_type, dtype=torch.long, device=device)
        return edge_index, edge_type

    def build_frequency_features(self, device: torch.device) -> torch.Tensor:
        # Frequency feature aligned to IRES paper intent: d_i + sum(freq(r) for outgoing r from i)
        relation_freq = Counter()
        for rel, idx in self.relation_to_idx.items():
            relation_freq[idx] = 0
        for r in self.edge_type:
            relation_freq[r] += 1

        undirected_degree = Counter()
        outgoing_rel_sum = Counter()
        for s, o, r in zip(self.edge_src, self.edge_dst, self.edge_type):
            undirected_degree[s] += 1
            undirected_degree[o] += 1
            outgoing_rel_sum[s] += relation_freq[r]

        n_nodes = len(self.idx_to_entity)
        x = torch.zeros((n_nodes, 1), dtype=torch.float32, device=device)
        for node_idx in range(n_nodes):
            x[node_idx, 0] = float(undirected_degree[node_idx] + outgoing_rel_sum[node_idx])
        return x


class IRESRGCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = FastRGCNConv(hidden_channels, hidden_channels, num_relations)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_type)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_type)
        h = self.bn2(h)
        return h


def modularity_trace_loss(
    z: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, device: torch.device
) -> torch.Tensor:
    # Sparse-safe formulation that avoids sparse@sparse autograd instability on some torch builds.
    # Tr(C^T A C) = sum(C * (A C))
    m = max(edge_index.shape[1] / 2.0, 1.0)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=device)

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg = deg / m

    a_z = torch.sparse.mm(adj, z)
    zt_a_z_trace = torch.sum(z * a_z)

    d_z = torch.mm(deg.unsqueeze(0), z)
    zt_dd_z_trace = torch.sum(d_z * d_z)

    trace_val = zt_a_z_trace - zt_dd_z_trace
    return trace_val / (4.0 * m)


def rank_entity_triples(
    entity: str,
    outgoing: Dict[str, List[Triple]],
    entity_to_idx: Dict[str, int],
    sim: torch.Tensor,
    k: int,
) -> List[Triple]:
    triples = outgoing.get(entity, [])
    if not triples:
        return []

    scored: List[Tuple[float, Triple]] = []
    src_idx = entity_to_idx.get(entity)
    if src_idx is None:
        return []

    for t in triples:
        _, p, o = t
        dst_idx = entity_to_idx.get(o)
        if dst_idx is None:
            continue
        score = float(sim[src_idx, dst_idx].item())
        scored.append((score, t))

    scored.sort(key=lambda item: (-item[0], item[1][1], item[1][2]))
    return [triple for _, triple in scored[:k]]


def format_summary_triple(triple: Triple) -> str:
    s, p, o = triple
    return f"<{s}> <{p}> \"{denormalize_entity(o)}\" ."


def write_outputs(
    output_dir: Path,
    rows: List[Dict],
    builder: M3GQAGraphBuilder,
    sim: torch.Tensor,
    ks: List[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = "datasets/M3GQA/all_single_setting/evaluation_subset.jsonl"

    for k in ks:
        out_path = output_dir / f"ires_m3gqa_train_top{k}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record_index, _ in enumerate(rows):
                entities = builder.topic_entities_by_record[record_index]
                outgoing = builder.outgoing_by_record[record_index]

                summary: List[str] = []
                for entity in entities:
                    topk = rank_entity_triples(
                        entity=entity,
                        outgoing=outgoing,
                        entity_to_idx=builder.entity_to_idx,
                        sim=sim,
                        k=k,
                    )
                    summary.extend(format_summary_triple(t) for t in topk)

                result = {
                    "meta": {
                        "mode": "dataset",
                        "dataset_json": dataset_json,
                        "record_index": record_index,
                        "graph_id": builder.graph_id_by_record[record_index],
                        "top_k": k,
                    },
                    "output": {
                        "record_index": record_index,
                        "question": builder.question_by_record[record_index],
                        "entities": [f"<{e}>" for e in entities],
                        "summary": summary,
                    },
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} records -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IRES-style RGCN on M3GQA and output top-k summaries.")
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path(
            "/home/asepfajarfirmansyah/Documents/GitHub/DL4MES/datasets/M3GQA/all_single_setting/evaluation_subset.jsonl"
        ),
        help="Path to M3GQA evaluation_subset.jsonl",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path(
            "/home/asepfajarfirmansyah/Documents/GitHub/DL4MES/datasets/M3GQA/all_single_setting/graphs"
        ),
        help="Directory with graph_{graph_id}.nt files; when present these triples are used for training",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs_train"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="RGCN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--step-size", type=int, default=7, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.01, help="StepLR gamma")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5], help="Top-k summary sizes")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of rows to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Training device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = list(load_jsonl(args.dataset_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found in {args.dataset_jsonl}")

    builder = M3GQAGraphBuilder()
    for idx, row in enumerate(rows):
        graph_id = int(row.get("graph_id", -1))
        graph_file = args.graphs_dir / f"graph_{graph_id}.nt"
        graph_edges = load_graph_nt_edges(graph_file)
        # Prefer richer graph triples from graphs dir; fallback to row['edges'] if file missing/empty.
        builder.add_record(idx, row, edges_override=graph_edges if graph_edges else None)

    edge_index, edge_type = builder.build_tensors(device)
    x = builder.build_frequency_features(device)

    model = IRESRGCNEncoder(
        in_channels=x.shape[1],
        hidden_channels=args.hidden_dim,
        num_relations=max(len(builder.relation_to_idx), 1),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        z = model(x, edge_index, edge_type)
        loss = modularity_trace_loss(z, edge_index, num_nodes=x.shape[0], device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        z = model(x, edge_index, edge_type)
        sim = torch.mm(z, z.t()).cpu()

    write_outputs(args.output_dir, rows, builder, sim, args.ks)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "entity_to_idx": builder.entity_to_idx,
            "relation_to_idx": builder.relation_to_idx,
            "args": vars(args),
        },
        args.output_dir / "ires_m3gqa_model.pt",
    )
    print(f"Saved model -> {args.output_dir / 'ires_m3gqa_model.pt'}")


if __name__ == "__main__":
    main()
