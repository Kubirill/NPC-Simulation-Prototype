# npc_benchmark_run.py
import importlib.util
import sys
import random
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def run_one(params):
    import importlib.util, sys
    from pathlib import Path
    mod_path = Path("npc_sim_prototype.py")
    spec = importlib.util.spec_from_file_location("npc_sim_prototype", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["npc_sim_prototype"] = mod
    spec.loader.exec_module(mod)
    cities, edges, npcs, steps, seed = params

    stats, events, meta = mod.run_simulation(
        n_cities=cities,
        edges_per_node=edges,
        n_npcs=npcs,
        steps=steps,
        seed=seed,
    )
    df = pd.DataFrame(stats._records)
    bdm_ms = df.loc[df["name"]=="build_distance_matrix", "duration_ms"].sum()
    trio_ms = df.loc[df["name"].isin(["select_npc","update_priorities","shaker_sort"]), "duration_ms"].sum()
    trio_per_step_ms = trio_ms / steps if steps>0 else np.nan
    return dict(
        n_cities=cities,
        edges_per_node=edges,
        n_npcs=npcs,
        build_distance_matrix_ms=bdm_ms,
        trio_total_ms=trio_ms,
        trio_per_step_ms=trio_per_step_ms,
    )

if __name__ == "__main__":
    # === Настройки ===
    STEPS = 200
    CITY_RANGE = range(50, 1001, 50)
    NPC_RANGE = range(50, 1001, 50)
    EDGES_RANGE = range(1, 11)
    SEED = 42
    N_CORES = multiprocessing.cpu_count()

    rng = random.Random(SEED)
    params_list = [
        (cities, edges, npcs, STEPS, rng.randrange(10_000_000))
        for cities in CITY_RANGE
        for edges in EDGES_RANGE
        for npcs in NPC_RANGE
    ]

    print(f"Всего комбинаций: {len(params_list)}")
    start = time()

    results = []
    with ProcessPoolExecutor(max_workers=N_CORES) as ex:
        futures = [ex.submit(run_one, p) for p in params_list]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Ошибка: {e}")
            if i % 50 == 0:
                print(f"Готово {i}/{len(params_list)}")

    print(f"Время выполнения: {(time()-start)/60:.1f} мин")
    res = pd.DataFrame(results)
    res.to_csv("npc_benchmark_grid.csv", index=False)
    print("✅ Результаты сохранены в npc_benchmark_grid.csv")
