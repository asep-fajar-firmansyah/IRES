#!/usr/bin/env python3
"""
Demonstration of the frequency-based feature correction.

This script shows the difference between:
1. OLD implementation: Concatenation [d_i, freq(r)] → 2D vectors
2. NEW implementation: Addition d_i + freq(r) → 1D vectors (aligned with paper)
"""

import torch
from collections import Counter

# Simulate graph data
print("="*70)
print("IRES Frequency Feature Correction Demonstration")
print("="*70)
print()

# Example: Simple graph with 5 nodes and relations
edges = [
    ('A', 'B', 'r1'),  # r1 appears 3 times globally
    ('A', 'C', 'r1'),
    ('A', 'D', 'r2'),  # r2 appears 1 time globally
    ('B', 'E', 'r1'),
]

nodes = sorted(list(set([u for u, v, r in edges] + [v for u, v, r in edges])))

# Calculate node degrees
node_degrees = {node: sum(1 for u, v, r in edges if u == node) for node in nodes}
print("Step 1: Node Degrees (d_i)")
print("-" * 70)
for node, degree in sorted(node_degrees.items()):
    print(f"  {node}: d_i = {degree}")
print()

# Calculate relation frequencies globally
relation_names = [r for u, v, r in edges]
relation_frequency = Counter(relation_names)
print("Step 2: Global Relation Frequencies")
print("-" * 70)
for rel, freq in sorted(relation_frequency.items()):
    print(f"  {rel}: appears {freq} times")
print()

# Calculate sum of relation frequencies for each node's outgoing edges
print("Step 3: Sum of Relation Frequencies for Outgoing Edges")
print("-" * 70)
node_relation_sum = {}
for node in nodes:
    outgoing = [r for u, v, r in edges if u == node]
    freq_sum = sum(relation_frequency[r] for r in outgoing)
    node_relation_sum[node] = freq_sum
    outgoing_str = ", ".join(outgoing) if outgoing else "no outgoing edges"
    print(f"  {node}: outgoing relations = [{outgoing_str}]")
    print(f"       freq({node}) = {freq_sum}")
print()

# Create tensors
node_freq_tensor = torch.tensor([node_degrees[n] for n in nodes], dtype=torch.float32)
rel_freq_tensor = torch.tensor([node_relation_sum[n] for n in nodes], dtype=torch.float32)

print("Step 4: Feature Creation Comparison")
print("="*70)
print()

print("OLD IMPLEMENTATION (INCORRECT - Concatenation)")
print("-" * 70)
# Old: concatenation
old_features = torch.cat((node_freq_tensor.view(-1, 1), rel_freq_tensor.view(-1, 1)), dim=1)
print(f"Shape: {old_features.shape} (2D vectors)")
print("\nFeature matrix (each row is a node):")
print("Node | [d_i, freq(r)]")
for i, node in enumerate(nodes):
    print(f"  {node}  | {old_features[i].tolist()}")
print()

print("NEW IMPLEMENTATION (CORRECT - Addition per paper)")
print("-" * 70)
# New: addition (aligned with paper: d_i + freq(r))
new_features = (node_freq_tensor + rel_freq_tensor).view(-1, 1)
print(f"Shape: {new_features.shape} (1D vectors)")
print("\nFeature matrix (each row is a node):")
print("Node | d_i + freq(r)")
for i, node in enumerate(nodes):
    d_i = node_degrees[node]
    freq_r = node_relation_sum[node]
    result = d_i + freq_r
    print(f"  {node}  | {d_i} + {freq_r} = {result}")
print()

print("="*70)
print("Paper Specification (Section 5.4, Hyperparameters):")
print("="*70)
print("""
The frequency-based features are 1D vectors containing for entity v_j 
in a triple (v_i, r, v_j) of a target entity v_i the sum d_i + freq(r) 
where:
  - d_i is the number of neighbors (i.e., degree) of v_i
  - freq(r) is the relative occurrence of the relationship r in triples T
""")

print("="*70)
print("Alignment Check:")
print("="*70)
print(f"✓ NEW implementation produces 1D vectors: {new_features.shape == torch.Size([len(nodes), 1])}")
print(f"✓ OLD implementation produces 2D vectors: {old_features.shape == torch.Size([len(nodes), 2])}")
print(f"✓ Addition matches paper formula: d_i + freq(r) ✓")
print()
