"""
NPC Simulation Prototype (Windows-friendly, no external deps)

Implements a prototype of the method from the paper "Метод реализации
псевдореалистичного перемещения NPC" (2025) with:

1) Random graph generator of N cities (nodes) and routes (weighted edges).
2) All-pairs distance matrix computation (NxN) using repeated Dijkstra.
3) Generation of X NPCs with random parameters.
4) Multiple-player-move simulation:
   - Player jumps to a random city after a random time delta.
   - Algorithm chooses which NPC can be encountered now (priority + reachability).
   - Priorities updated (selected NPC: delay; others: approach increment).
   - NPC list re-sorted by priority using Shaker (Cocktail) sort.
5) Detailed per-function timing + CSV export.

API
---
You can import and call `run_simulation(...)` from another script, or run
this module as a CLI to produce CSV files with timings and events.

Example (CLI):
    python npc_sim_prototype.py --cities 200 --npcs 150 --steps 5000 \
        --edges-per-node 3 --seed 42 --out-prefix run1

Example (import):
    from npc_sim_prototype import run_simulation
    stats, events, meta = run_simulation(
        n_cities=200,
        n_npcs=150,
        steps=1000,
        edges_per_node=3,
        seed=123,
    )
    stats.save_csv("timings.csv")
    stats.save_events_csv("events.csv")

Notes
-----
- Only the Python standard library is used.
- Works on Windows without Docker.
- The distance matrix is computed by running Dijkstra from each source node.
- The NPC selection obeys:  distance(current, last_seen) <= speed * (t_now - t_last)
  and requires positive priority; the first (in priority order) that satisfies is chosen.
- After each selection, priorities are updated and then re-sorted via Shaker sort.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple

# ---------------------------
# Utility: timing collector
# ---------------------------
class StatsCollector:
    def __init__(self) -> None:
        self._records: List[dict] = []
        self._events: List[dict] = []
        self._counter: int = 0

    class _Timer:
        def __init__(self, outer: "StatsCollector", name: str, **meta):
            self.outer = outer
            self.name = name
            self.meta = meta
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            t1 = time.perf_counter()
            rec = {"name": self.name, "duration_ms": (t1 - self.t0) * 1000.0}
            rec.update(self.meta)
            rec["ts"] = time.time()
            self.outer._records.append(rec)

    def timeit(self, name: str, **meta):
        return StatsCollector._Timer(self, name, **meta)

    def log_event(self, **fields):
        fields = dict(fields)
        fields.setdefault("event_id", self._counter)
        self._counter += 1
        self._events.append(fields)

    # ---- export ----
    def save_csv(self, filename: str) -> None:
        if not self._records:
            return
        keys = sorted({k for r in self._records for k in r.keys()})
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._records)

    def save_events_csv(self, filename: str) -> None:
        if not self._events:
            return
        keys = sorted({k for r in self._events for k in r.keys()})
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._events)

    def summary(self) -> Dict[str, Dict[str, float]]:
        from statistics import mean
        groups: Dict[str, List[float]] = {}
        for r in self._records:
            groups.setdefault(r["name"], []).append(r["duration_ms"])
        out: Dict[str, Dict[str, float]] = {}
        for k, vals in groups.items():
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            p95 = vals_sorted[min(n - 1, int(0.95 * n))] if n else 0.0
            out[k] = {
                "count": float(n),
                "total_ms": float(sum(vals_sorted)),
                "avg_ms": float(mean(vals_sorted)) if n else 0.0,
                "p95_ms": float(p95),
                "max_ms": float(vals_sorted[-1]) if n else 0.0,
            }
        return out

# ---------------------------
# Graph
# ---------------------------
class Graph:
    def __init__(self, n: int):
        self.n = n
        # adjacency: node -> List[(neighbor, weight)]
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    def add_undirected_edge(self, u: int, v: int, w: float) -> None:
        if u == v:
            return
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    @staticmethod
    def generate(n: int, edges_per_node: int = 3, min_w: float = 1.0, max_w: float = 50.0, rng: random.Random | None = None) -> "Graph":
        """Generate a connected undirected weighted graph with ~edges_per_node.
        Ensures connectivity by first creating a random spanning tree, then adds
        extra random edges.
        """
        rng = rng or random
        g = Graph(n)
        # 1) Random spanning tree
        order = list(range(n))
        rng.shuffle(order)
        for i in range(1, n):
            u = order[i]
            v = order[rng.randrange(0, i)]  # connect to any previous
            w = rng.uniform(min_w, max_w)
            g.add_undirected_edge(u, v, w)
        # 2) Add extra edges to approach target avg degree
        target_edges = max(0, int((edges_per_node * n) // 2) - (n - 1))
        added = 0
        attempts = 0
        while added < target_edges and attempts < 10 * target_edges + 1000:
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                attempts += 1
                continue
            # avoid duplicate edges by a quick check
            if any(nei == v for (nei, _) in g.adj[u]):
                attempts += 1
                continue
            w = rng.uniform(min_w, max_w)
            g.add_undirected_edge(u, v, w)
            added += 1
        return g

    def dijkstra(self, src: int) -> List[float]:
        n = self.n
        dist = [math.inf] * n
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]
        while pq:
            d, u = heappop(pq)
            if d != dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(pq, (nd, v))
        return dist

    def all_pairs_distance_matrix(self, stats: Optional[StatsCollector] = None) -> List[List[float]]:
        n = self.n
        M = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            with (stats.timeit("dijkstra", src=i) if stats else _null_timer()):
                dist = self.dijkstra(i)
            for j in range(n):
                M[i][j] = dist[j]
        return M

class _null_timer:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False

# ---------------------------
# NPCs and selection
# ---------------------------
@dataclass
class NPC:
    npc_id: int
    speed: float                  # V_npc
    priority: float               # P
    local_priority: float         # tie-breaker
    last_area: int                # l (ID of last seen area)
    last_seen_time: float         # t_NPC
    delay_after_meet: float       # P_s (subtract on meet)
    approach_increment: float     # P_a (add to others)

    def can_reach(self, current_area: int, current_time: float, M: List[List[float]]) -> bool:
        # Eq. S <= V * Δt ; S = M[current][last_area]; Δt = t_now - t_last
        S = M[current_area][self.last_area]
        dt = current_time - self.last_seen_time
        if dt < 0:
            return False
        return S <= (self.speed * dt)

# Shaker (Cocktail) sort: in-place, descending by (priority, local_priority)
# Provided to measure exactly this sort variant.
def shaker_sort_npcs(npcs: List[NPC]) -> None:
    n = len(npcs)
    if n < 2:
        return
    left, right = 0, n - 1
    while left < right:
        new_right = left
        for i in range(left, right):
            if (npcs[i].priority < npcs[i+1].priority) or (
                npcs[i].priority == npcs[i+1].priority and npcs[i].local_priority < npcs[i+1].local_priority):
                npcs[i], npcs[i+1] = npcs[i+1], npcs[i]
                new_right = i
        right = new_right
        new_left = right
        for i in range(right, left, -1):
            if (npcs[i-1].priority < npcs[i].priority) or (
                npcs[i-1].priority == npcs[i].priority and npcs[i-1].local_priority < npcs[i].local_priority):
                npcs[i-1], npcs[i] = npcs[i], npcs[i-1]
                new_left = i
        left = new_left

# Select first reachable NPC with positive priority in already priority-sorted list.
def select_npc(current_area: int, current_time: float, npcs: List[NPC], M: List[List[float]]) -> Optional[NPC]:
    for npc in npcs:
        if npc.priority <= 0:
            # array is sorted; everything beyond is <= 0
            return None
        if npc.can_reach(current_area, current_time, M):
            return npc
    return None

# Update priorities after encounter
def update_priorities(selected: Optional[NPC], npcs: List[NPC], current_area: int, current_time: float) -> None:
    if selected is None:
        return
    # Selected NPC: decrease priority, update last seen
    selected.priority -= selected.delay_after_meet
    selected.last_area = current_area
    selected.last_seen_time = current_time
    # Others: increase priority
    for npc in npcs:
        if npc is not selected:
            npc.priority += npc.approach_increment

# ---------------------------
# Simulation
# ---------------------------
@dataclass
class SimulationConfig:
    n_cities: int = 100
    edges_per_node: int = 3
    n_npcs: int = 100
    steps: int = 1000
    # Graph weights
    min_edge_w: float = 1.0
    max_edge_w: float = 50.0
    # NPC params ranges
    min_speed: float = 1.0
    max_speed: float = 10.0
    min_priority0: float = 0.0
    max_priority0: float = 10.0
    min_local_priority0: float = 0.0
    max_local_priority0: float = 1.0
    min_delay_after_meet: float = 5.0
    max_delay_after_meet: float = 15.0
    min_approach_increment: float = 0.5
    max_approach_increment: float = 2.0
    # Player time delta per step
    dt_min: float = 1.0
    dt_max: float = 5.0
    # Initial last_seen_time sampled from [-t_init_spread, 0]
    t_init_spread: float = 50.0
    # Random seed
    seed: Optional[int] = None

@dataclass
class SimulationMeta:
    config: SimulationConfig
    distance_matrix_shape: Tuple[int, int]


def _rng_uniform(rng: random.Random, a: float, b: float) -> float:
    return rng.uniform(a, b)


def _gen_npcs(cfg: SimulationConfig, rng: random.Random) -> List[NPC]:
    npcs: List[NPC] = []
    for i in range(cfg.n_npcs):
        npc = NPC(
            npc_id=i,
            speed=_rng_uniform(rng, cfg.min_speed, cfg.max_speed),
            priority=_rng_uniform(rng, cfg.min_priority0, cfg.max_priority0),
            local_priority=_rng_uniform(rng, cfg.min_local_priority0, cfg.max_local_priority0),
            last_area=rng.randrange(cfg.n_cities),
            last_seen_time=-_rng_uniform(rng, 0.0, cfg.t_init_spread),
            delay_after_meet=_rng_uniform(rng, cfg.min_delay_after_meet, cfg.max_delay_after_meet),
            approach_increment=_rng_uniform(rng, cfg.min_approach_increment, cfg.max_approach_increment),
        )
        npcs.append(npc)
    return npcs


def run_simulation(
    n_cities: int = 100,
    edges_per_node: int = 3,
    n_npcs: int = 100,
    steps: int = 1000,
    min_edge_w: float = 1.0,
    max_edge_w: float = 50.0,
    min_speed: float = 1.0,
    max_speed: float = 10.0,
    min_priority0: float = 0.0,
    max_priority0: float = 10.0,
    min_local_priority0: float = 0.0,
    max_local_priority0: float = 1.0,
    min_delay_after_meet: float = 5.0,
    max_delay_after_meet: float = 15.0,
    min_approach_increment: float = 0.5,
    max_approach_increment: float = 2.0,
    dt_min: float = 1.0,
    dt_max: float = 5.0,
    t_init_spread: float = 50.0,
    seed: Optional[int] = None,
) -> Tuple[StatsCollector, List[dict], SimulationMeta]:
    """
    Run the full simulation once.

    Returns:
        stats: StatsCollector with per-function timings
        events: list of step-level events (player moves + selected npc)
        meta:  summary info
    """
    cfg = SimulationConfig(
        n_cities=n_cities,
        edges_per_node=edges_per_node,
        n_npcs=n_npcs,
        steps=steps,
        min_edge_w=min_edge_w,
        max_edge_w=max_edge_w,
        min_speed=min_speed,
        max_speed=max_speed,
        min_priority0=min_priority0,
        max_priority0=max_priority0,
        min_local_priority0=min_local_priority0,
        max_local_priority0=max_local_priority0,
        min_delay_after_meet=min_delay_after_meet,
        max_delay_after_meet=max_delay_after_meet,
        min_approach_increment=min_approach_increment,
        max_approach_increment=max_approach_increment,
        dt_min=dt_min,
        dt_max=dt_max,
        t_init_spread=t_init_spread,
        seed=seed,
    )

    rng = random.Random(seed)
    stats = StatsCollector()

    # 1) Graph generation
    with stats.timeit("generate_graph", n_cities=cfg.n_cities, edges_per_node=cfg.edges_per_node):
        g = Graph.generate(cfg.n_cities, cfg.edges_per_node, cfg.min_edge_w, cfg.max_edge_w, rng)

    # 2) All-pairs distance matrix
    with stats.timeit("build_distance_matrix", n=cfg.n_cities):
        M = g.all_pairs_distance_matrix(stats=None)  # we can also time each Dijkstra by passing stats

    # 3) NPCs
    with stats.timeit("create_npcs", n_npcs=cfg.n_npcs):
        npcs = _gen_npcs(cfg, rng)
    with stats.timeit("initial_sort_shaker", n_npcs=cfg.n_npcs):
        shaker_sort_npcs(npcs)

    # 4) Simulation
    events: List[dict] = []
    current_time = 0.0
    for step in range(cfg.steps):
        # player move
        dt = _rng_uniform(rng, cfg.dt_min, cfg.dt_max)
        current_time += dt
        current_area = rng.randrange(cfg.n_cities)

        # select NPC
        with stats.timeit("select_npc", step=step):
            selected = select_npc(current_area, current_time, npcs, M)

        # update priorities
        with stats.timeit("update_priorities", step=step):
            update_priorities(selected, npcs, current_area, current_time)

        # re-sort by priority via shaker sort
        with stats.timeit("shaker_sort", step=step):
            shaker_sort_npcs(npcs)

        # log event
        ev = {
            "step": step,
            "t": current_time,
            "dt": dt,
            "area": current_area,
            "selected_npc": (selected.npc_id if selected else None),
        }
        events.append(ev)
        stats.log_event(**ev)

    meta = SimulationMeta(config=cfg, distance_matrix_shape=(len(M), len(M[0]) if M else 0))
    return stats, events, meta


# ---------------------------
# CLI
# ---------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NPC pseudo-realistic movement simulation prototype")
    p.add_argument("--cities", type=int, default=100)
    p.add_argument("--edges-per-node", type=int, default=3)
    p.add_argument("--npcs", type=int, default=100)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dt-min", type=float, default=1.0)
    p.add_argument("--dt-max", type=float, default=5.0)
    p.add_argument("--min-speed", type=float, default=1.0)
    p.add_argument("--max-speed", type=float, default=10.0)
    p.add_argument("--min-priority0", type=float, default=0.0)
    p.add_argument("--max-priority0", type=float, default=10.0)
    p.add_argument("--min-local-priority0", type=float, default=0.0)
    p.add_argument("--max-local-priority0", type=float, default=1.0)
    p.add_argument("--min-delay-after-meet", type=float, default=5.0)
    p.add_argument("--max-delay-after-meet", type=float, default=15.0)
    p.add_argument("--min-approach-increment", type=float, default=0.5)
    p.add_argument("--max-approach-increment", type=float, default=2.0)
    p.add_argument("--min-edge-w", type=float, default=1.0)
    p.add_argument("--max-edge-w", type=float, default=50.0)
    p.add_argument("--t-init-spread", type=float, default=50.0)
    p.add_argument("--out-prefix", type=str, default="npc_run")
    p.add_argument("--no-files", action="store_true", help="Do not write CSV files, just print a summary")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    stats, events, meta = run_simulation(
        n_cities=args.cities,
        edges_per_node=args.edges_per_node,
        n_npcs=args.npcs,
        steps=args.steps,
        min_edge_w=args.min_edge_w,
        max_edge_w=args.max_edge_w,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        min_priority0=args.min_priority0,
        max_priority0=args.max_priority0,
        min_local_priority0=args.min_local_priority0,
        max_local_priority0=args.max_local_priority0,
        min_delay_after_meet=args.min_delay_after_meet,
        max_delay_after_meet=args.max_delay_after_meet,
        min_approach_increment=args.min_approach_increment,
        max_approach_increment=args.max_approach_increment,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        t_init_spread=args.t_init_spread,
        seed=args.seed,
    )

    summary = stats.summary()
    print("=== Timing Summary (ms) ===")
    for name, s in summary.items():
        print(f"{name:24s} count={int(s['count']):6d} avg={s['avg_ms']:.3f} p95={s['p95_ms']:.3f} max={s['max_ms']:.3f} total={s['total_ms']:.3f}")

    if not args.no_files:
        ts = int(time.time())
        timings_path = f"{args.out_prefix}_timings_{ts}.csv"
        events_path = f"{args.out_prefix}_events_{ts}.csv"
        stats.save_csv(timings_path)
        stats.save_events_csv(events_path)
        print(f"Saved: {timings_path}\nSaved: {events_path}")

    # small meta echo
    print(f"Matrix shape: {meta.distance_matrix_shape}, steps={args.steps}, cities={args.cities}, npcs={args.npcs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
