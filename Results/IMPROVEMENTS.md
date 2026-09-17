# BP-MDS: solver improvements (September 2026)

Goal: improve BP-MDS by at least 20% in cost or time on million-scale instances without
degrading the other. Result: on every instance tried the new solver is both faster and
cheaper. On the synthetic 1M instance it is 2.9x faster and 15% cheaper; on Lazio (FILO2,
1M customers) it is 13.9x faster and 0.2% cheaper.

All runs: WSL2 Ubuntu on a Ryzen 7 7435HS (8 cores / 16 threads), g++ 15.2, `-O3 -march=native -flto -fopenmp`.
Time is the solver's own "Execution time for solving" from the `.sol` header. Every solution
passed `Solution::verify` and, independently, `Scripts/verify_solution.py` (a standalone Python
checker that re-parses the `.vrp` and `.sol`, checks coverage and capacity, and recomputes the
cost; agreement is within 1e-12 relative on all files).

## Million-scale results

| Instance | N | alpha | rho | Baseline time | New time | Speedup | Baseline cost | New cost | Cost change |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| XML1000000_1173_01 (synthetic, random) | 1,000,001 | 0.3 | 1000 | 39.4 s | 13.0 to 13.8 s | 2.9x | 1,221,080,092 | 1,035,200,485 | -15.2% |
| XML1000000_2235_01 (synthetic, clustered) | 1,000,001 | 0.3 | 1000 | 42.9 s | 12.9 s | 3.3x | 403,656,501 | 399,886,485 | -0.9% |
| Lazio (FILO2, BKS 3,145,381,332) | 1,000,000 | 5 | 1000 | 349.7 s | 25.2 s | 13.9x | 3,190,791,324 (gap 1.44%) | 3,185,188,717 (gap 1.27%) | -0.2% |
| Lazio, dense MST (`BPMDS_SPARSE_MST=100000000`) | 1,000,000 | 5 | 1000 | 349.7 s | 36.6 s | 9.6x | same | 3,181,604,811 (gap 1.15%) | -0.3% |

Notes.
- Baseline cost varies with `random_device`; two baseline runs on the synthetic instance gave 1,221,080,092 and 1,220,671,450. New-solver runs with seeds 1, 2, 3 gave 1,035,473,054 / 1,035,200,485 / 1,035,274,446, so the gap is far outside run-to-run noise.
- rho = 10000 on the synthetic instance: 35.7 s, cost 1,035,503,337. The local search now dominates solution quality, so the large rho used in the original experiments no longer buys cost. The old solver at rho = 10000 would take roughly 400 s.
- The synthetic instance needs 130,386 routes instead of 154,795: the optimal split packs vehicles fuller, which is where most of the 15% comes from. Lazio has demands 1, 2, 3 in equal thirds (total 1,998,931, capacity 50), so the lower bound is 39,979 routes; the new solution uses 40,422, about 1.1% above the bound, so almost nothing is left to gain from packing and its win is time.
- Peak memory rises (synthetic 20 MB to 32 MB, Lazio 49 MB to 228 MB) because each bucket keeps a few candidate route sets and neighbour lists alive; still negligible next to the 1M-node instance itself.

## Smaller CVRPLIB instances (sanity check against best known solutions, rho 1000, repo's best alpha)

| Instance | alpha | BKS | Baseline cost (gap) | New cost (gap) |
|:--|--:|--:|--:|--:|
| X-n101-k25 | 360 | 27,591 | 30,826 (11.7%) | 28,938 (4.9%) |
| X-n1001-k43 | 80 | 72,355 | 79,976 (10.5%) | 77,312 (6.9%) |
| Antwerp1 | 18 | 477,277 | 513,728 (7.6%) | 499,279 (4.6%) |
| Flanders1 | 18 | 7,240,118 | 7,675,401 (6.0%) | 7,507,456 (3.7%) |

## Where the time went (baseline)

At million scale the repo uses alpha between 0.1 and 0.5 degrees, i.e. 700 to 3600 buckets of a few
hundred to a few thousand customers, and rho of 5000 to 10000. Per bucket the baseline did:

1. Prim's MST with a binary heap over the complete graph, calling `CVRP::distance` (two bounds checks per call) on global ids: O(n^2 log n) with poor locality.
2. rho randomized DFS passes, each allocating a `new[]` per visited node, constructing a `std::random_device` and `std::mt19937`, building the route vectors, and copying the whole route set under an `omp critical` whenever a better cost appeared.
3. A 2-opt that rebuilt and re-summed the whole tour for every (i, k) pair: O(n^3) per improvement.
4. Nested `parallel for` (buckets x rho) with `omp_set_nested`, oversubscribing 16 cores with up to 256 threads (25 s of system time on a 40 s run).
5. Bucket assignment testing every sector per customer (O(N x buckets)).

## What changed

Time.
- Buckets are solved in bucket-local index space with contiguous x, y, demand arrays (`Bucket_Workspace`). No bounds-checked global lookups in the hot loops.
- Dense array Prim (no heap) for buckets up to 4000 nodes. On a complete Euclidean graph it is the same O(n^2) but vectorises and touches memory sequentially.
- Buckets above 4000 nodes (Lazio at alpha 5 has 14k per bucket) use Kruskal over the 32-nearest-neighbour graph plus depot edges, stitching any leftover components. Lazio's MST phase went from 131 to 3.6 CPU seconds; the tree is not always the exact Euclidean MST, which moved Lazio's cost by +0.11% relative to dense Prim (the synthetic instance moved -0.03%). `BPMDS_SPARSE_MST` sets the threshold.
- The DFS is a template with no allocation: a CSR copy of the tree is shuffled in place per visited node with a splitmix64 generator, the stack is two flat arrays, and the pass only accumulates cost. The ordering is fully determined by a seed derived from (global seed, bucket id, iteration), so the winning orderings are regenerated once instead of storing routes for every iteration.
- One level of parallelism: buckets across threads (dynamic schedule, largest first) when there are at least as many buckets as threads, otherwise rho across threads. No nesting, no oversubscription.
- O(N) bucket assignment via `atan2`. This also fixes a latent bug: a customer coincident with the depot produced a NaN unit vector and fell into no bucket.
- Input parsing with `strtol`/`strtod` instead of a `stringstream` per line (outside the timed region, but halves wall-clock on 1M lines). Coordinates are now parsed as double, not float.

Cost.
- Optimal split (Prins) of the DFS giant tour instead of the greedy cut: O(n x route length) dynamic programme that is never worse than the greedy partition of the same sequence. Applied to the 16 best orderings by greedy cost.
- Intra-route local search: 2-opt plus Or-opt (segments of 1 to 3, both orientations) with O(1) delta evaluation, replacing the O(n^3) 2-opt.
- Inter-route local search inside each bucket on the 4 best candidates after split: relocate, swap and 2-opt* (tail exchange) between routes, candidates limited to the 32 nearest neighbours (grid based k-NN), routes as circular doubly linked lists, a work queue with don't-look bits and reverse-neighbour wake-ups so only customers near a change are re-examined.
- The original per-route polish (nearest-neighbour reorder + 2-opt, keep the cheapest of three) still runs last, so nothing the old pipeline did is lost.

## Knobs

- `--seed=<uint64>`: reproducible runs (default: random).
- Environment: `BPMDS_K` (orderings that get split, 16), `BPMDS_M` (of those, how many get inter-route search, 4), `BPMDS_KNN` (neighbour list size, 32), `BPMDS_WORK` (inter-route work bound as multiple of bucket size, 50), `BPMDS_SPARSE_MST` (bucket size above which the k-NN MST is used, 4000), `BPMDS_PROFILE=1` (per-phase CPU seconds on stderr).

Sweep on the synthetic 1M instance (alpha 0.3, rho 1000, seed 1) that fixed the defaults:

| K | M | KNN | time | cost |
|--:|--:|--:|--:|--:|
| 16 | 16 | 10 | 13.2 s | 1,105,333,068 |
| 16 | 4 | 16 | 9.9 s | 1,072,092,301 |
| 16 | 8 | 16 | 15.3 s | 1,067,619,458 |
| 16 | 4 | 24 | 13.6 s | 1,048,140,034 |
| 16 | 4 | 32 | 16.1 s | 1,035,473,054 |
| 16 | 16 | 16 | 25.7 s | 1,063,702,117 |

Neighbourhood width matters more than the number of candidates.

## What was tried and dropped

- Running the full inter-route search on all 16 candidates with full-pass scanning: cost 1,101,007,104 but 293 CPU seconds in that phase alone (22 s wall). The queue version with reverse-neighbour wake-ups reaches the same quality at a fraction of the work.
- A first k-NN grid walk that visited every cell inside each ring instead of only the ring's perimeter: cubic in the grid size, 45 s on a 7-node instance whose bucket bounding box was degenerate.

## Not done (honest list)

- Buckets never exchange customers, so routes at wedge borders are never improved across the border. A boundary pass between adjacent buckets is the next cost lever.
- Or-opt and 2-opt* are first-improvement; no perturbation / restarts, so each bucket stops at the first local optimum.
- Memory: candidate route sets for K orderings are kept alive per bucket; could be streamed.
- All timings are from one machine (WSL2, 16 threads); cluster numbers will differ but the phase profile (`BPMDS_PROFILE=1`) shows where time goes.

## Reproduce

```bash
make
# synthetic 1M: python3 Scripts/Inputs-generation/generator.py 1000000 1 1 7 3 1 42   (in Inputs/Synthetic)
./Bin/bucket-partitioned-MDS --alpha=0.3 --rho=1000 --seed=1 --input=Inputs/Synthetic/XML1000000_1173_01.vrp --output=out.sol
./Bin/bucket-partitioned-MDS --alpha=5 --rho=1000 --seed=1 --input=Inputs/FILO2/I/Lazio.vrp --output=lazio.sol
python3 Scripts/verify_solution.py Inputs/FILO2/I/Lazio.vrp lazio.sol
```
