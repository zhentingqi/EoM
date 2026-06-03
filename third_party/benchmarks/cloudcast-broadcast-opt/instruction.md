# CloudCast — Multi-Cloud Broadcast Optimization

You are evolving a single Python file, `initial_program.py`, that defines:

```python
def search_algorithm(
    src: str,            # source cloud region, e.g. "aws:us-east-1"
    dsts: list[str],     # destination regions
    G: nx.DiGraph,       # network graph; per-edge attrs: cost ($/GB), throughput (Gbps)
    num_partitions: int, # number of data partitions to broadcast
) -> BroadCastTopology:
    ...
```

The function builds a `BroadCastTopology` describing, for each destination and each
partition, the chain of edges that data flows through.  The verifier then runs a
network simulator on five scenarios (intra-AWS / intra-Azure / intra-GCP /
inter-cloud AGZ / inter-cloud GAZ2) and computes the total egress cost.

## Score

```
score = combined_score = 1 / (1 + total_cost)
```

Higher is better.  The seed (Dijkstra single-path) gets `total_cost ≈ 1035`,
i.e. `score ≈ 0.000965`.  Cutting `total_cost` in half roughly doubles the
score.

## Workspace

Only the file `initial_program.py` is read by the verifier.  You **must keep**:

- the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers,
- the `BroadCastTopology` and `make_nx_graph` definitions inside the block,
- the `search_algorithm` function name and signature.

Helper functions / new classes inside the EVOLVE-BLOCK are fine.  Do not
import unavailable third-party packages — only the Python stdlib plus
`networkx` and `pandas` are guaranteed to be present.

## Validation

The simulator validates the returned topology before scoring.  Every
`(dst, partition)` pair must have a non-empty edge list, edges must be
present in `G`, and the path must be continuous from `src` to `dst`.
A topology that fails validation scores `0`.

### Edge format — read this carefully

Every entry in `bc.paths[dst][str(p)]` must be a **3-element list**
`[src_node, dst_node, edge_data]` where `edge_data` is the *per-edge*
attribute dict — not the whole graph.  Use `G[u][v]` (a dict view), not
`G` (the DiGraph), as the third element:

```python
# correct
bc.append_dst_partition_path(dst, j, [u, v, G[u][v]])

# WRONG — len OK but the simulator's __construct_g will crash
bc.append_dst_partition_path(dst, j, [u, v, G])

# WRONG — len < 3, validator rejects
bc.append_dst_partition_path(dst, j, [u, v])
bc.append_dst_partition_path(dst, j, (u, v))
```

Path continuity is enforced: for partition `p` reaching `dst`, the first
edge must start at `src`, each subsequent edge must start where the
previous ended, and the last edge must end at `dst`.

### `G` is a `networkx.DiGraph` — use the nx API, not dict API

`G` is **not** a plain dict.  Common mistakes that crash the verifier:

```python
G.get(u, {})            # ❌ AttributeError: DiGraph has no 'get'
G[u].get(v)             # ❌ AdjacencyView has no 'get' on older nx
```

Use one of:

```python
G[u][v]                                # raises KeyError if missing
G.get_edge_data(u, v, default={})      # safe lookup
G.has_edge(u, v)                       # boolean test
G.edges[u, v]["cost"]                  # attribute read
nx.dijkstra_path(G, src, dst, weight="cost")
nx.shortest_simple_paths(G, src, dst, weight="cost")  # generator of K-shortest
```

Per-edge attrs live in `G[u][v]` as a dict-like view with at least
`"cost"` (USD/GB) and `"throughput"` (Gbps).

## How to iterate

The Hayek runtime gives you a `request_eval()` tool that runs the verifier
mid-episode.  The score *delta* against your previous checkpoint is the
reward you collect — improvements pay, regressions do not.  Plan small
edits, eval, then iterate.

Strategy hints (you do not have to follow these):

- Multi-path broadcast: split each partition across cheap parallel routes.
- Use `G[u][v]["cost"]` as the primary weight; `throughput` constrains
  effective flow.
- Avoid sending the same partition over the same edge twice; the simulator
  charges per partition × edge.
