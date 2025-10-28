# Быстрый старт

## Требования

- Python 3.8+

Пакеты:

```
pandas
numpy
matplotlib
scikit-learn
```

Установка:

```bash
python -m venv .venv
# Unix / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn
```

> Примечание: сам модуль симуляции (`npc_sim_prototype.py`) использует только стандартную библиотеку и может запускаться без внешних зависимостей.

---

# Запуск

## Одиночная симуляция (CLI)

Пример запуска:

```bash
python npc_sim_prototype.py \
  --cities 200 \
  --npcs 150 \
  --steps 5000 \
  --edges-per-node 3 \
  --seed 42 \
  --out-prefix run1
```

Основные параметры:

- `--cities` — число областей/узлов (матрица расстояний будет `cities x cities`).
- `--npcs` — число NPC.
- `--steps` — число шагов/итераций симуляции.
- `--edges-per-node` — плотность/связность графа (сколько ребер на узел при генерации).
- `--seed` — seed для повторяемости.
- `--out-prefix` — префикс для файлов вывода (тайминги/события).
- `--no-files` — выводить только сводку в консоль, не сохранять CSV.

После выполнения (без `--no-files`) появятся CSV-файлы, например:

- `<out-prefix>_timings_<ts>.csv` — тайминги по этапам/функциям.
- `<out-prefix>_events_<ts>.csv` — список/лог событий симуляции.

## Бенчмарк (grid) — `simulation.py`

Запускает серию симуляций по сетке параметров (параллельно, с использованием `multiprocessing`).

```bash
python simulation.py
```

Результат: `npc_benchmark_grid.csv` с агрегированными метриками по каждой комбинации параметров.



## Построение графиков — `npc_benchmark_plot.py`

Ожидает файл `npc_benchmark_grid.csv` в текущей директории.

```bash
python npc_benchmark_plot.py
```




# Выводы / структура файлов

- `*_timings_*.csv` — таблица таймингов по ключевым этапам симуляции.
- `*_events_*.csv` — хронологический лог событий (опционально сохраняется).
- `npc_benchmark_grid.csv` — агрегированный результат бенчмарка по сетке параметров.
- `*.png` — графики, построенные `npc_benchmark_plot.py`.

