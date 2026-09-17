# BP-MDS meeting cheat sheet

Keep this open during the call. Do not try to explain every code line. For every point use:

> **Problem → change → why it helps → evidence → limitation**

---

## 30-second opening

> BP-MDS solves a capacitated vehicle-routing problem at million-customer scale. It divides customers into angular buckets around the depot, builds a tree in each bucket, generates randomized DFS customer orders, splits those orders into truck routes, and improves the routes locally. I first profiled the million-scale path, then removed unnecessary work from the repeated DFS samples, improved the route construction and local search, and verified the output independently.

---

## Results to state accurately

Same `alpha` and `rho`, 16 threads, WSL2 laptop. Every generated solution was checked independently.

| Instance | Baseline time | New time | Cost change |
|---|---:|---:|---:|
| Synthetic random, 1M | 39.4 s | 13.0 s | 15.2% lower |
| Synthetic clustered, 1M | 42.9 s | 12.9 s | 0.9% lower |
| Lazio / FILO2, 1M | 349.7 s | 25.2 s | BKS gap 1.44% → 1.27% |

The requested target was **20% improvement in time or cost without hurting the other**. Time improved by 3× to 14×, and cost improved in every reported test.

**Do not claim:** these exact seconds will be identical on another machine or cluster.

---

## 1. Main speed change: make the rho loop cheap

### What is rho?

`rho` is the number of random DFS orders tried **per bucket**.

```text
rho = 1000
→ try 1000 different random customer orders in one bucket
→ calculate a route cost for each
→ keep promising ones
```

With many buckets, the number of DFS attempts becomes very large. This was the hottest part of the solver.

### Old problem

For every random DFS attempt, old code:

```text
1. allocated neighbour memory while walking the tree
2. created full route vectors
3. allocated/deallocated traversal state
4. discarded all this work if that candidate lost
```

The old code had repeated `new node_t[...]` and `delete[]` inside the DFS traversal.

### Change

```text
All rho attempts:
    do the randomized DFS
    calculate only its greedy cost
    keep its attempt index / reproducible seed

Best 16 attempts:
    replay the same DFS exactly
    construct actual routes
    run split and local search
```

The reusable arrays are allocated once per bucket and reset for each DFS attempt.

### Why it helps

We still try the same number of randomized DFS orders. We only stop creating expensive full route objects for losing samples.

```text
Old: 1000 samples → 1000 full route constructions
New: 1000 samples → 1000 cheap cost calculations → 16 route constructions
```

### Say this

> The first major bottleneck was the rho loop. The original code constructed full routes and repeatedly allocated and freed memory for every random DFS sample, although most samples were later discarded. I kept the same number of samples, but made each sample calculate only its cost. Because the walk is reproducible from a seed, I replay only the best 16 candidates and construct routes for those. So the search is not reduced; the unnecessary bookkeeping is removed.

### Exact code

- `Lib/Bucket_Partitioned_MDS/Solver.cpp`, `random_dfs()`, around line 241.
- Same file, scoring phase and replay phase, around lines 1037–1085.

### Likely questions

**Did you reduce rho?**  
No. I kept rho the same; I made a losing sample cheaper.

**How can you reconstruct a route later?**  
The seed plus bucket ID plus attempt ID regenerates the same randomized DFS ordering.

**Why 16?**  
It was a measured quality/time tradeoff. It is configurable by `BPMDS_K`.

**Can sample 17 become best after local search?**  
Yes, in theory. Top-16 selection is a heuristic, not a proof. Tuning showed it was a good balance.

**Did all allocation disappear?**  
No. Repeated allocation in the hot DFS loop was removed. The solver still keeps bucket scratch storage and selected candidate routes.

---

## 2. Parallelism: use CPU threads once, not in nested loops

### Old problem

There were two places to parallelize:

```text
different buckets
different rho trials inside one bucket
```

The old code put OpenMP parallel loops around both. Nested OpenMP can add scheduling overhead; depending on configuration, inner parallel regions can either oversubscribe threads or be serialized.

### Change

```text
If there are many buckets:
    parallelize buckets
    keep rho loop serial inside each bucket

If there are few large buckets:
    solve bucket(s) serially
    parallelize rho trials inside a bucket
```

Largest buckets run first with dynamic scheduling, so a worker does not get stuck with the final large bucket while others sit idle.

### Say this

> I made the solver choose one parallel level based on where there is enough work. With many buckets, threads solve buckets. With few large buckets, threads evaluate rho samples inside a bucket. This keeps CPU utilization predictable and avoids relying on inefficient nested OpenMP behavior.

### Exact code

- `Lib/Bucket_Partitioned_MDS/Solver.cpp`, `Solver::solve()`, around line 1143.

---

## 3. Optimal split: main route-cost improvement

### Old problem: greedy cutting

DFS produces one long customer order:

```text
A → B → C → D → E
```

The old code fills a truck until the next customer does not fit, then closes the route immediately. That is a greedy cut.

```text
Truck 1: A, B
Truck 2: C, D
Truck 3: E
```

This is valid, but it is not necessarily the cheapest way to cut the same order into truck routes.

### Change

Use dynamic programming to test all capacity-valid route endings and choose the cheapest set of cuts for the same order.

```text
Greedy split = one valid partition of the order
Optimal split = cheapest valid partition of that same order
```

### Why it helps

It packs variable customer demands into trucks more intelligently and can reduce unnecessary depot returns. On the random synthetic instance, routes fell from about 155k to 130k.

Lazio benefited less because each customer has demand 1 and vehicle capacity is 50; routes are already almost exactly full.

### Say this

> I kept the DFS customer order fixed, but replaced the greedy capacity cut with an exact dynamic-programming split of that order. Since greedy is one feasible split, the optimal split cannot be worse for that order. This was especially useful when demands vary because it packs vehicles more fully.

### Exact code

- `Lib/Bucket_Partitioned_MDS/Solver.cpp`, `optimal_split()`, around line 340.

---

## 4. Local search: improve the good route candidates

After split, the solution is feasible but may have unnecessary travel. Local search makes small improving changes.

### Within one route

- **2-opt:** reverse a route segment to remove crossings.
- **Or-opt:** move a chain of 1–3 customers to a better place in the same route.

### Between routes in the same bucket

- **Relocate:** move one customer to another route.
- **Swap:** exchange customers between routes.
- **2-opt\*:** cut two routes and exchange their tails.

### Why it remains fast

The solver considers only a customer's **32 nearest neighbours**, found by a spatial grid. It does not compare every customer with every other customer, which would be far too slow at million scale.

### Say this

> I added a bounded local search after construction. It uses standard relocate, swap, and 2-opt-star moves across routes, plus 2-opt and Or-opt within routes. To preserve scale, each customer considers only 32 geographically nearest neighbours rather than all customers.

### Important nuance

This is a heuristic. It stops at a local optimum; it does not guarantee the global optimum.

---

## 5. Sparse MST for huge buckets

### Problem

For a big Lazio bucket, dense Prim MST construction compares many point pairs. This is roughly quadratic work and became costly at around 14k nodes per bucket.

### Change

For buckets above 4000 nodes, build a sparse graph from each customer's 32 nearest neighbours plus depot edges, then run Kruskal-style MST construction on that graph.

### Tradeoff

The sparse tree may not be the exact Euclidean MST. It made Lazio much faster, with a small quality tradeoff versus dense MST, but the total result remained better than the original baseline on both time and cost. It can be turned off using `BPMDS_SPARSE_MST`.

### Say this

> For very large buckets, I reused the nearest-neighbour graph already needed by local search to avoid dense quadratic MST work. This is a controlled speed-quality tradeoff, and I kept it configurable because the exact MST is not always reproduced.

---

## 6. Other smaller but useful changes

- Direct bucket assignment with `atan2`: calculate a customer's angle once and place it directly in its sector.
- Bucket-local contiguous coordinate/demand arrays: improve cache locality and avoid repeated global object access in hot loops.
- Fast deterministic RNG (`splitmix64`): cheap random branching and reproducible runs.
- `--seed=<number>`: same seed gives the same run.
- `BPMDS_PROFILE=1`: prints phase CPU time to identify bottlenecks.
- Guard against a single customer's demand exceeding vehicle capacity.

---

## Validation: show this if asked

Build:

```bash
make
```

Run a sample:

```bash
./Bin/bucket-partitioned-MDS \
  --alpha=30 --rho=100 --seed=1 \
  --input=Inputs/Sample/toy.vrp \
  --output=Results/Output/Sample/toy.sol
```

Independently verify it:

```bash
python3 Scripts/verify_solution.py \
  Inputs/Sample/toy.vrp \
  Results/Output/Sample/toy.sol
```

The checker verifies:

```text
every customer appears exactly once
every route stays within vehicle capacity
the total route distance is recomputed independently
```

Expected ending:

```text
OK
```

Say:

> The checker is separate from the solver. It parses the instance and output file itself, checks coverage and capacity, and recomputes distance rather than trusting the solver's printed cost.

---

## Honest limitations / next version

Say these clearly; they make you sound careful.

1. **No cross-bucket repair.** Customers near a wedge border cannot move into a neighbouring bucket's route. This is the clearest next cost improvement.
2. **Local optimum only.** The local search does not yet use perturbation and restarts.
3. **CPU only.** GPU implementation is future work.
4. **More memory.** Neighbour lists and candidate routes increased peak memory (Lazio roughly 49 MB → 228 MB).
5. **Single-machine timings.** Relative phase bottlenecks should transfer, but exact seconds need cluster reproduction.

### Best answer: “What would you do next?”

> First I would add a border-repair pass between adjacent angular buckets using relocate, swap, and 2-opt-star moves. Then I would add a perturb-and-reoptimize loop to escape local optima. The randomized DFS trials and neighbour-based move evaluation are also natural GPU candidates because much of that work is independent.

---

## Last-minute answers

**Did you make it faster by searching less?**  
No. I used the same rho random DFS samples; I removed route construction and repeated allocation for losing samples.

**Why does optimal split never hurt for a fixed order?**  
The greedy cut is one feasible split. Dynamic programming chooses the cheapest feasible split among all of them.

**Why only 32 neighbours?**  
Most useful route improvements are geographically local. A fixed neighbour list avoids all-pairs work and keeps the search scalable.

**How do you know the solution is correct?**  
Independent checker: coverage, capacity, and independently recomputed total distance.

**What is the key lesson?**  
At million scale, removing unnecessary repeated work can matter more than adding a more complicated algorithm.
