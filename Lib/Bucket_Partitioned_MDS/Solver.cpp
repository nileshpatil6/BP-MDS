#include "Utils.h"
#include "Bucket_Partitioned_MDS.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <omp.h>

namespace Bucket_Partitioned_MDS
{
    namespace
    {
        /*
        * Fast_RNG: splitmix64. Deterministic given its seed, so a DFS ordering can be
        * reproduced later from (seed) alone without storing the routes it generated.
        */
        struct Fast_RNG
        {
            std::uint64_t s;
            explicit Fast_RNG(std::uint64_t seed) : s(seed) {}

            inline std::uint64_t next()
            {
                std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                return z ^ (z >> 31);
            }

            // Uniform integer in [0, n)
            inline std::uint32_t bounded(std::uint32_t n)
            {
                return (std::uint32_t)(((next() >> 32) * (std::uint64_t)n) >> 32);
            }
        };

        inline std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b, std::uint64_t c)
        {
            Fast_RNG r(a ^ (b * 0x9e3779b97f4a7c15ULL) ^ (c * 0xbf58476d1ce4e5b9ULL));
            return r.next();
        }

        inline int env_int(const char* name, int def)
        {
            const char* v = std::getenv(name);
            return v ? std::atoi(v) : def;
        }
        // Number of best DFS orderings (by greedy cost) that get the optimal split treatment
        const int SPLIT_CANDIDATES = env_int("BPMDS_K", 16);
        // Of those, how many get the inter-route local search
        const int INTER_CANDIDATES = env_int("BPMDS_M", 4);
        // Neighbour list size for the inter-route local search
        const int KNN = env_int("BPMDS_KNN", 32);
        // Work bound for the inter-route search, as a multiple of bucket size
        const int INTER_WORK_FACTOR = env_int("BPMDS_WORK", 50);
        // Buckets larger than this use the k-NN sparse MST instead of dense Prim
        const int SPARSE_MST_MIN_NODES = env_int("BPMDS_SPARSE_MST", 4000);

        // Optional phase profiling (BPMDS_PROFILE=1): CPU seconds summed over all threads
        enum Phase { P_MST, P_DFS, P_KNN, P_SPLIT, P_INTRA, P_INTER, P_FINAL, P_COUNT };
        const char* PHASE_NAME[P_COUNT] = { "mst", "dfs", "knn", "split", "intra_ls", "inter_ls", "final_polish" };
        double phase_time[P_COUNT] = { 0 };
        inline void add_phase(Phase p, double t)
        {
            #pragma omp atomic
            phase_time[p] += t;
        }
    }

    struct Bucket_Workspace
    {
        /*
        * Bucket_Workspace: Everything one bucket needs, in bucket-local index space.
        * Coordinates and demands are copied into contiguous arrays so that MST
        * construction and the rho DFS passes stay cache friendly and never touch
        * the global CVRP object (which bounds-checks every access).
        */
        int n = 0;
        capacity_t Q = 0;

        std::vector <cord_t>    x, y;
        std::vector <demand_t>  dem;
        std::vector <distance_t> d0;   // distance of each local node to depot (local 0)

        // MST in CSR form: neighbours of u are adj[off[u] .. off[u+1])
        std::vector <int> off;
        std::vector <int> adj;

        // Per DFS scratch
        std::vector <int>  adj_scratch;   // shuffled copy of adj
        std::vector <char> visited;
        std::vector <int>  stack_node;
        std::vector <int>  stack_pos;
        std::vector <int>  seq;           // DFS order of customers (local ids)

        // Split DP scratch
        std::vector <distance_t> dp;
        std::vector <int>        pred;

        void init(const CVRP& cvrp, const std::vector <node_t>& bucket)
        {
            n = bucket.size();
            Q = cvrp.capacity();
            x.resize(n); y.resize(n); dem.resize(n); d0.resize(n);
            for (int i = 0; i < n; i++)
            {
                const Point& p = cvrp[bucket[i]];
                x[i] = p.x; y[i] = p.y; dem[i] = p.demand;
            }
            for (int i = 0; i < n; i++)
            {
                d0[i] = dist(0, i);
            }
            off.assign(n + 1, 0);
            adj.assign(2 * (n - 1), 0);
            adj_scratch.assign(2 * (n - 1), 0);
            visited.assign(n, 0);
            stack_node.assign(n, 0);
            stack_pos.assign(n, 0);
            seq.assign(n, 0);
            dp.assign(n + 1, 0);
            pred.assign(n + 1, 0);
        }

        inline distance_t dist(int a, int b) const
        {
            const cord_t dx = x[a] - x[b];
            const cord_t dy = y[a] - y[b];
            return std::sqrt(dx * dx + dy * dy);
        }
    };

    void Solver::create_buckets(
        const CVRP& cvrp,
        std::vector <std::vector <node_t>>& buckets) const
    {
        /*
        * create_buckets: Assign every customer to the angular sector [i*alpha, (i+1)*alpha)
        * measured counter clockwise from the positive x axis at the depot. O(N) via atan2
        * instead of testing every sector per customer.
        */
        const int num_buckets = buckets.size();
        const int N = cvrp.size();
        const node_t depot = cvrp.depot();

        // Depot goes into all buckets
        for (int i = 0; i < num_buckets; i++)
        {
            buckets[i].push_back(depot);
        }

        if (num_buckets == 1)
        {
            buckets[0].reserve(N);
            for (node_t u = 1; u < N; u++)
            {
                buckets[0].push_back(u);
            }
            return;
        }

        std::vector <int> bucket_of(N, 0);
        const Point& dp = cvrp[depot];

        #pragma omp parallel for schedule(static)
        for (node_t u = 1; u < N; u++)
        {
            const Point& p = cvrp[u];
            double ang = std::atan2(p.y - dp.y, p.x - dp.x) * 180.0 / PI;
            if (ang < 0) ang += 360.0;
            int idx = (int)std::floor(ang / alpha);
            if (idx >= num_buckets) idx = num_buckets - 1;
            if (idx < 0) idx = 0;
            bucket_of[u] = idx;
        }

        std::vector <int> count(num_buckets, 0);
        for (node_t u = 1; u < N; u++) count[bucket_of[u]]++;
        for (int i = 0; i < num_buckets; i++) buckets[i].reserve(count[i] + 1);
        for (node_t u = 1; u < N; u++) buckets[bucket_of[u]].push_back(u);
    }

    void Solver::construct_mst(
        Bucket_Workspace& ws) const
    {
        /*
        * construct_mst: Dense Prim on the bucket (complete Euclidean graph). O(n^2) with
        * flat arrays, which beats a binary heap on a complete graph and vectorises.
        * Result is written to ws.off / ws.adj (CSR adjacency of the tree).
        */
        const int n = ws.n;
        std::vector <distance_t> key(n, DBL_MAX);
        std::vector <int> parent(n, -1);
        std::vector <char> in_tree(n, 0);

        key[0] = 0;
        int u = 0;
        for (int it = 0; it < n; it++)
        {
            in_tree[u] = 1;
            const cord_t ux = ws.x[u], uy = ws.y[u];
            distance_t best = DBL_MAX;
            int best_v = -1;
            for (int v = 0; v < n; v++)
            {
                if (in_tree[v]) continue;
                const cord_t dx = ws.x[v] - ux;
                const cord_t dy = ws.y[v] - uy;
                const distance_t d = std::sqrt(dx * dx + dy * dy);
                if (d < key[v]) { key[v] = d; parent[v] = u; }
                if (key[v] < best) { best = key[v]; best_v = v; }
            }
            if (best_v < 0) break;
            u = best_v;
        }

        // Degrees then CSR
        std::fill(ws.off.begin(), ws.off.end(), 0);
        for (int v = 1; v < n; v++)
        {
            ws.off[v + 1]++;
            ws.off[parent[v] + 1]++;
        }
        for (int v = 0; v < n; v++) ws.off[v + 1] += ws.off[v];
        std::vector <int> fill(ws.off.begin(), ws.off.end() - 1);
        for (int v = 1; v < n; v++)
        {
            const int p = parent[v];
            ws.adj[fill[v]++] = p;
            ws.adj[fill[p]++] = v;
        }
    }

    namespace
    {
        /*
        * random_dfs: One randomized DFS over the MST starting at the depot (local 0).
        * Children of each node are visited in a random order drawn from `seed`.
        * The customers are chained in visitation order and cut greedily whenever
        * the next customer does not fit in the vehicle. Returns the greedy cost.
        * If record is true, the visitation order (customers only) is stored in ws.seq.
        * No heap allocation happens here.
        */
        template <bool RECORD>
        distance_t random_dfs(Bucket_Workspace& ws, std::uint64_t seed)
        {
            const int n = ws.n;
            const int* off = ws.off.data();
            int* adj = ws.adj_scratch.data();
            char* visited = ws.visited.data();
            int* st_node = ws.stack_node.data();
            int* st_pos = ws.stack_pos.data();
            const demand_t* dem = ws.dem.data();

            std::memcpy(adj, ws.adj.data(), sizeof(int) * ws.adj.size());
            std::memset(visited, 0, n);

            Fast_RNG rng(seed);
            auto shuffle_children = [&](int u)
            {
                const int b = off[u], e = off[u + 1];
                for (int i = e - 1; i > b; i--)
                {
                    const int j = b + (int)rng.bounded((std::uint32_t)(i - b + 1));
                    std::swap(adj[i], adj[j]);
                }
            };

            distance_t cost = 0;
            capacity_t residue = ws.Q;
            int prev = 0;
            int seq_len = 0;

            visited[0] = 1;
            shuffle_children(0);
            st_node[0] = 0;
            st_pos[0] = off[0];
            int sp = 1;

            while (sp > 0)
            {
                const int u = st_node[sp - 1];
                int p = st_pos[sp - 1];
                const int e = off[u + 1];
                while (p < e && visited[adj[p]]) p++;
                if (p == e)
                {
                    sp--;
                    continue;
                }
                const int v = adj[p];
                st_pos[sp - 1] = p + 1;
                visited[v] = 1;

                const demand_t d = dem[v];
                if (residue < d)
                {
                    cost += ws.d0[prev];
                    residue = ws.Q;
                    prev = 0;
                }
                cost += (prev == 0) ? ws.d0[v] : ws.dist(prev, v);
                residue -= d;
                prev = v;
                if (RECORD) ws.seq[seq_len++] = v;

                shuffle_children(v);
                st_node[sp] = v;
                st_pos[sp] = off[v];
                sp++;
            }

            if (prev != 0) cost += ws.d0[prev];
            return cost;
        }

        /*
        * optimal_split: Given the giant tour ws.seq[0..m-1], choose the cut points that
        * minimise total route length subject to capacity (Prins' split, O(m * L)).
        * The greedy cut used during the DFS is one feasible partition of this tour, so
        * the result is never worse than the greedy cost. Writes routes (local ids).
        */
        distance_t optimal_split(
            Bucket_Workspace& ws,
            std::vector <std::vector <int>>& routes)
        {
            const int m = ws.n - 1;
            const int* seq = ws.seq.data();
            distance_t* dp = ws.dp.data();
            int* pred = ws.pred.data();

            dp[0] = 0;
            for (int i = 1; i <= m; i++)
            {
                dp[i] = DBL_MAX;
                capacity_t load = 0;
                distance_t path = 0;
                const distance_t tail = ws.d0[seq[i - 1]];
                for (int j = i - 1; j >= 0; j--)
                {
                    load += ws.dem[seq[j]];
                    if (load > ws.Q) break;
                    if (j < i - 1) path += ws.dist(seq[j], seq[j + 1]);
                    const distance_t cand = dp[j] + ws.d0[seq[j]] + path + tail;
                    if (cand < dp[i]) { dp[i] = cand; pred[i] = j; }
                }
            }

            // Reconstruct
            int nroutes = 0;
            for (int i = m; i > 0; i = pred[i]) nroutes++;
            routes.clear();
            routes.resize(nroutes);
            int r = nroutes - 1;
            for (int i = m; i > 0; i = pred[i], r--)
            {
                const int j = pred[i];
                routes[r].reserve(i - j);
                for (int k = j; k < i; k++) routes[r].push_back(seq[k]);
            }
            return dp[m];
        }
    }

    distance_t Solver::get_route_distance(
        const CVRP&                 cvrp,
        const std::vector <node_t>& route) const
    {
        node_t prev_node = cvrp.depot();
        distance_t cost = 0;
        for (auto& node : route)
        {
            cost += cvrp.distance(prev_node, node);
            prev_node = node;
        }
        cost += cvrp.distance(prev_node, cvrp.depot());
        return cost;
    }

    void Solver::tsp_approx(
        const CVRP&             cvrp,
        std::vector<node_t>&    cities,
        std::vector<node_t>&    tour,
        const int               ncities) const
    {
        /*
        * tsp_approx: Nearest neighbour ordering starting at the depot (cities.back()).
        */
        node_t ClosePt = 0;
        distance_t CloseDist;

        std::copy(cities.begin(), cities.end() - 1, tour.begin() + 1);
        tour[0] = cities.back();

        for (int i = 1; i < ncities; i++)
        {
            distance_t ThisX = cvrp[tour[i - 1]].x;
            distance_t ThisY = cvrp[tour[i - 1]].y;
            CloseDist = DBL_MAX;

            for (int j = ncities - 1;; j--)
            {
                distance_t ThisDist = (cvrp[tour[j]].x - ThisX) * (cvrp[tour[j]].x - ThisX);
                if (ThisDist <= CloseDist)
                {
                    ThisDist += (cvrp[tour[j]].y - ThisY) * (cvrp[tour[j]].y - ThisY);
                    if (ThisDist <= CloseDist)
                    {
                        if (j < i) break;
                        CloseDist = ThisDist;
                        ClosePt = j;
                    }
                }
            }
            std::swap(tour[i], tour[ClosePt]);
        }
    }

    std::vector<std::vector<node_t>> Solver::process_tsp_approx(
        const CVRP&                             cvrp,
        const std::vector<std::vector<node_t>>& sol_routes) const
    {
        std::vector<std::vector<node_t>> modifiedRoutes;
        const int num_routes = sol_routes.size();
        modifiedRoutes.reserve(num_routes);

        for (int i = 0; i < num_routes; i++)
        {
            int sz = sol_routes[i].size();
            std::vector<node_t> cities = sol_routes[i];
            cities.push_back(cvrp.depot());
            std::vector<node_t> tour(sz + 1);
            tsp_approx(cvrp, cities, tour, sz + 1);
            modifiedRoutes.emplace_back(tour.begin() + 1, tour.begin() + 1 + sz);
        }
        return modifiedRoutes;
    }

    namespace
    {
        /*
        * path_local_search: 2-opt (segment reversal) and Or-opt (move a segment of 1..3
        * customers, either orientation) on the open path r[0] -> r[1..m] -> r[m+1], where
        * r[0] and r[m+1] are depot slots. O(1) delta evaluation, first improvement,
        * repeated until no move improves. Coordinates are looked up through `px`/`py`.
        */
        void path_local_search(const cord_t* px, const cord_t* py, std::vector <int>& r)
        {
            const int m = (int)r.size() - 2;
            if (m < 2) return;
            auto D = [&](int a, int b) -> distance_t
            {
                const cord_t dx = px[a] - px[b], dy = py[a] - py[b];
                return std::sqrt(dx * dx + dy * dy);
            };
            const distance_t EPS_MOVE = 1e-9;

            bool improved = true;
            while (improved)
            {
                improved = false;

                for (int i = 1; i <= m - 1; i++)
                {
                    const int a = r[i - 1], b = r[i];
                    for (int k = i + 1; k <= m; k++)
                    {
                        const int c = r[k], d = r[k + 1];
                        const distance_t delta = D(a, c) + D(b, d) - D(a, b) - D(c, d);
                        if (delta < -EPS_MOVE)
                        {
                            std::reverse(r.begin() + i, r.begin() + k + 1);
                            improved = true;
                            break; // r[i] changed: move on to the next i with fresh endpoints
                        }
                    }
                }

                for (int L = 1; L <= 3 && L < m; L++)
                {
                    for (int i = 1; i + L - 1 <= m; i++)
                    {
                        const int s = r[i], e = r[i + L - 1];
                        const int prev = r[i - 1], next = r[i + L];
                        const distance_t remove_gain = D(prev, s) + D(e, next) - D(prev, next);
                        for (int j = 0; j <= m; j++)
                        {
                            if (j >= i - 1 && j <= i + L - 1) continue;
                            const int p = r[j], q = r[j + 1];
                            const distance_t base = D(p, q);
                            const distance_t ins_fwd = D(p, s) + D(e, q) - base;
                            const distance_t ins_rev = D(p, e) + D(s, q) - base;
                            const bool rev = ins_rev < ins_fwd;
                            const distance_t delta = (rev ? ins_rev : ins_fwd) - remove_gain;
                            if (delta < -EPS_MOVE)
                            {
                                int seg[3];
                                for (int t = 0; t < L; t++) seg[t] = r[i + t];
                                if (rev) std::reverse(seg, seg + L);
                                r.erase(r.begin() + i, r.begin() + i + L);
                                int pos = (j < i) ? j + 1 : j + 1 - L;
                                r.insert(r.begin() + pos, seg, seg + L);
                                improved = true;
                                break; // segment moved: continue with the next i
                            }
                        }
                    }
                }
            }
        }

        /*
        * build_knn: k nearest customers of every customer inside the bucket, via a uniform
        * grid (expected O(n k)). Result: knn[u * k + j], padded with -1.
        */
        void build_knn(const Bucket_Workspace& ws, int k, std::vector <int>& knn)
        {
            const int n = ws.n;
            knn.assign((size_t)n * k, -1);
            if (n <= 2) return;

            cord_t minx = ws.x[1], maxx = ws.x[1], miny = ws.y[1], maxy = ws.y[1];
            for (int i = 2; i < n; i++)
            {
                minx = std::min(minx, ws.x[i]); maxx = std::max(maxx, ws.x[i]);
                miny = std::min(miny, ws.y[i]); maxy = std::max(maxy, ws.y[i]);
            }
            const double w = std::max(maxx - minx, 1e-9), h = std::max(maxy - miny, 1e-9);
            // about 2 points per cell, grid shaped like the bounding box
            const int target_cells = std::max(1, (n - 1) / 2);
            int gx = (int)std::lround(std::sqrt(target_cells * w / h));
            gx = std::min(std::max(gx, 1), target_cells);
            int gy = std::min(std::max(target_cells / gx, 1), target_cells);
            const double cw = w / gx, ch = h / gy;

            auto cell_x = [&](cord_t x) { int c = (int)((x - minx) / cw); return std::min(std::max(c, 0), gx - 1); };
            auto cell_y = [&](cord_t y) { int c = (int)((y - miny) / ch); return std::min(std::max(c, 0), gy - 1); };

            std::vector <int> cell_off(gx * gy + 1, 0), cell_pts(n - 1), cell_of(n);
            for (int i = 1; i < n; i++) { cell_of[i] = cell_y(ws.y[i]) * gx + cell_x(ws.x[i]); cell_off[cell_of[i] + 1]++; }
            for (int c = 0; c < gx * gy; c++) cell_off[c + 1] += cell_off[c];
            std::vector <int> fill(cell_off.begin(), cell_off.end() - 1);
            for (int i = 1; i < n; i++) cell_pts[fill[cell_of[i]]++] = i;

            std::vector <std::pair <distance_t, int>> best;
            best.reserve(64);
            for (int u = 1; u < n; u++)
            {
                best.clear();
                const int cx = cell_of[u] % gx, cy = cell_of[u] / gx;
                distance_t kth = DBL_MAX;
                for (int ring = 0; ring <= std::max(gx, gy); ring++)
                {
                    // Nothing outside this ring can beat the current k-th best
                    if ((int)best.size() >= k && ring > 0)
                    {
                        const double reach = (ring - 1) * std::min(cw, ch);
                        if (reach * reach >= kth) break;
                    }
                    const int x0 = cx - ring, x1 = cx + ring, y0 = cy - ring, y1 = cy + ring;
                    if (x0 < 0 && y0 < 0 && x1 >= gx && y1 >= gy) break; // ring fully outside the grid
                    for (int yy = std::max(y0, 0); yy <= std::min(y1, gy - 1); yy++)
                    {
                        const bool edge_row = (yy == y0 || yy == y1);
                        // Perimeter only: whole row on the top/bottom edge, two cells otherwise
                        const int xa = edge_row ? std::max(x0, 0) : x0;
                        const int xb = edge_row ? std::min(x1, gx - 1) : x1;
                        const int step = edge_row ? 1 : std::max(x1 - x0, 1);
                        for (int xx = xa; xx <= xb; xx += step)
                        {
                            if (xx < 0 || xx >= gx) continue;
                            const int c = yy * gx + xx;
                            for (int p = cell_off[c]; p < cell_off[c + 1]; p++)
                            {
                                const int v = cell_pts[p];
                                if (v == u) continue;
                                const cord_t dx = ws.x[v] - ws.x[u], dy = ws.y[v] - ws.y[u];
                                const distance_t d2 = dx * dx + dy * dy;
                                if ((int)best.size() < k)
                                {
                                    best.push_back({d2, v});
                                    if ((int)best.size() == k)
                                    {
                                        std::sort(best.begin(), best.end());
                                        kth = best.back().first;
                                    }
                                }
                                else if (d2 < kth)
                                {
                                    best.back() = {d2, v};
                                    for (int t = k - 1; t > 0 && best[t] < best[t - 1]; t--) std::swap(best[t], best[t - 1]);
                                    kth = best.back().first;
                                }
                            }
                        }
                    }
                    if (x0 <= 0 && y0 <= 0 && x1 >= gx - 1 && y1 >= gy - 1) break;
                }
                if ((int)best.size() < k) std::sort(best.begin(), best.end());
                for (size_t j = 0; j < best.size(); j++) knn[(size_t)u * k + j] = best[j].second;
            }
        }

        /*
        * sparse_mst: spanning tree over the k-NN graph (plus depot-to-everyone edges) with
        * Kruskal, then any remaining components are stitched with their nearest outside
        * node. O(n k log(n k)) instead of the dense O(n^2) Prim; for k = 32 the result is
        * the Euclidean MST in practice. Used for large buckets only.
        */
        void sparse_mst(Bucket_Workspace& ws, const std::vector <int>& knn, int k)
        {
            const int n = ws.n;
            struct E { distance_t w; int u, v; };
            std::vector <E> edges;
            edges.reserve((size_t)n * k + n);
            for (int u = 1; u < n; u++)
                for (int j = 0; j < k; j++)
                {
                    const int v = knn[(size_t)u * k + j];
                    if (v < 0) break;
                    if (u < v) edges.push_back({ ws.dist(u, v), u, v });
                }
            for (int v = 1; v < n; v++) edges.push_back({ ws.d0[v], 0, v });
            std::sort(edges.begin(), edges.end(), [](const E& a, const E& b) { return a.w < b.w; });

            std::vector <int> uf(n);
            std::iota(uf.begin(), uf.end(), 0);
            auto find = [&](int a) { while (uf[a] != a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };

            std::vector <std::pair <int, int>> tree;
            tree.reserve(n - 1);
            for (const E& e : edges)
            {
                const int a = find(e.u), b = find(e.v);
                if (a == b) continue;
                uf[a] = b;
                tree.push_back({ e.u, e.v });
                if ((int)tree.size() == n - 1) break;
            }

            // Stitch leftover components (rare): smallest component to its nearest outsider
            while ((int)tree.size() < n - 1)
            {
                std::vector <int> root(n);
                std::vector <int> size(n, 0);
                for (int v = 0; v < n; v++) { root[v] = find(v); size[root[v]]++; }
                int small = -1;
                for (int v = 0; v < n; v++) if (size[v] > 0 && (small < 0 || size[v] < size[small])) small = v;
                distance_t best = DBL_MAX; int bu = -1, bv = -1;
                for (int u = 0; u < n; u++)
                {
                    if (root[u] != small) continue;
                    for (int v = 0; v < n; v++)
                    {
                        if (root[v] == small) continue;
                        const distance_t d = ws.dist(u, v);
                        if (d < best) { best = d; bu = u; bv = v; }
                    }
                }
                uf[find(bu)] = find(bv);
                tree.push_back({ bu, bv });
            }

            std::fill(ws.off.begin(), ws.off.end(), 0);
            for (auto& e : tree) { ws.off[e.first + 1]++; ws.off[e.second + 1]++; }
            for (int v = 0; v < n; v++) ws.off[v + 1] += ws.off[v];
            std::vector <int> fill(ws.off.begin(), ws.off.end() - 1);
            for (auto& e : tree)
            {
                ws.adj[fill[e.first]++] = e.second;
                ws.adj[fill[e.second]++] = e.first;
            }
        }

        /*
        * inter_route_search: relocate / swap / tail exchange (2-opt*) between routes of one
        * bucket, candidates restricted to k nearest neighbours. Routes are circular doubly
        * linked lists; route r has sentinel id n + r that stands for the depot.
        * Returns total cost of the routes after the search (routes rewritten, local ids).
        */
        distance_t inter_route_search(
            const Bucket_Workspace& ws,
            const std::vector <int>& knn, int k,
            std::vector <std::vector <int>>& routes)
        {
            const int n = ws.n;
            const int R = routes.size();
            const int T = n + R;
            std::vector <int> nxt(T), prv(T), rof(T);
            std::vector <capacity_t> load(R, 0);

            auto X = [&](int a) { return ws.x[a < n ? a : 0]; };
            auto Y = [&](int a) { return ws.y[a < n ? a : 0]; };
            auto D = [&](int a, int b) -> distance_t
            {
                const cord_t dx = X(a) - X(b), dy = Y(a) - Y(b);
                return std::sqrt(dx * dx + dy * dy);
            };

            for (int r = 0; r < R; r++)
            {
                const int s = n + r;
                int last = s;
                rof[s] = r;
                for (int v : routes[r])
                {
                    nxt[last] = v; prv[v] = last; rof[v] = r; load[r] += ws.dem[v];
                    last = v;
                }
                nxt[last] = s; prv[s] = last;
            }

            auto prefix_load = [&](int u) -> capacity_t
            {
                capacity_t l = 0;
                for (int a = u; a < n; a = prv[a]) l += ws.dem[a];
                return l;
            };

            const distance_t EPS_MOVE = 1e-9;

            // Work queue with "don't look" bits: a customer is re-examined only after a
            // move changed its own or a neighbour's surroundings.
            std::vector <int> queue;
            queue.reserve(4 * n);
            std::vector <char> queued(T, 0);
            auto push_one = [&](int a)
            {
                if (a < n && a > 0 && !queued[a]) { queued[a] = 1; queue.push_back(a); }
            };
            // Reverse neighbour lists: who has `a` in its candidate list. When `a` moves,
            // those customers may have gained an improving move too.
            std::vector <int> rk_off(n + 1, 0), rk;
            for (int u = 1; u < n; u++)
                for (int j = 0; j < k; j++) { const int v = knn[(size_t)u * k + j]; if (v < 0) break; rk_off[v + 1]++; }
            for (int v = 0; v < n; v++) rk_off[v + 1] += rk_off[v];
            rk.resize(rk_off[n]);
            {
                std::vector <int> fill(rk_off.begin(), rk_off.end() - 1);
                for (int u = 1; u < n; u++)
                    for (int j = 0; j < k; j++) { const int v = knn[(size_t)u * k + j]; if (v < 0) break; rk[fill[v]++] = u; }
            }
            auto push = [&](int a)
            {
                if (a >= n || a <= 0) return;
                push_one(a);
                for (int p = rk_off[a]; p < rk_off[a + 1]; p++) push_one(rk[p]);
            };
            for (int u = 1; u < n; u++) push(u);
            size_t head = 0;
            const size_t max_work = (size_t)INTER_WORK_FACTOR * n + 1000;
            size_t work = 0;

            while (head < queue.size() && work < max_work)
            {
                const int u = queue[head++];
                queued[u] = 0;
                work++;
                const int ru = rof[u];
                bool moved = false;
                for (int j = 0; j < k && !moved; j++)
                {
                    const int v = knn[(size_t)u * k + j];
                    if (v < 0) break;
                    const int rv = rof[v];
                    if (rv == ru) continue;
                    const int pu = prv[u], nu = nxt[u], pv = prv[v], nv = nxt[v];

                    // Relocate u next to v (after or before)
                    if (load[rv] + ws.dem[u] <= ws.Q)
                    {
                        const distance_t rem = D(pu, nu) - D(pu, u) - D(u, nu);
                        const distance_t ins_after = D(v, u) + D(u, nv) - D(v, nv);
                        const distance_t ins_before = D(pv, u) + D(u, v) - D(pv, v);
                        const bool after = ins_after <= ins_before;
                        const distance_t delta = rem + (after ? ins_after : ins_before);
                        if (delta < -EPS_MOVE)
                        {
                            nxt[pu] = nu; prv[nu] = pu;
                            if (after) { nxt[v] = u; prv[u] = v; nxt[u] = nv; prv[nv] = u; }
                            else       { nxt[pv] = u; prv[u] = pv; nxt[u] = v; prv[v] = u; }
                            load[ru] -= ws.dem[u]; load[rv] += ws.dem[u]; rof[u] = rv;
                            moved = true;
                        }
                    }

                    // Swap u and v
                    if (!moved && load[ru] - ws.dem[u] + ws.dem[v] <= ws.Q && load[rv] - ws.dem[v] + ws.dem[u] <= ws.Q)
                    {
                        const distance_t delta = D(pu, v) + D(v, nu) - D(pu, u) - D(u, nu)
                                               + D(pv, u) + D(u, nv) - D(pv, v) - D(v, nv);
                        if (delta < -EPS_MOVE)
                        {
                            nxt[pu] = v; prv[v] = pu; nxt[v] = nu; prv[nu] = v;
                            nxt[pv] = u; prv[u] = pv; nxt[u] = nv; prv[nv] = u;
                            load[ru] += ws.dem[v] - ws.dem[u]; load[rv] += ws.dem[u] - ws.dem[v];
                            rof[u] = rv; rof[v] = ru;
                            moved = true;
                        }
                    }

                    // 2-opt*: exchange the tails after u and after v
                    if (!moved)
                    {
                        const distance_t delta = D(u, nv) + D(v, nu) - D(u, nu) - D(v, nv);
                        if (delta < -EPS_MOVE)
                        {
                            const capacity_t pre_u = prefix_load(u), pre_v = prefix_load(v);
                            const capacity_t suf_u = load[ru] - pre_u, suf_v = load[rv] - pre_v;
                            if (pre_u + suf_v <= ws.Q && pre_v + suf_u <= ws.Q)
                            {
                                const int su = n + ru, sv = n + rv;
                                const int last_u = prv[su], last_v = prv[sv];
                                if (nv == sv) { nxt[u] = su; prv[su] = u; }
                                else
                                {
                                    nxt[u] = nv; prv[nv] = u;
                                    nxt[last_v] = su; prv[su] = last_v;
                                    for (int a = nv; a != su; a = nxt[a]) rof[a] = ru;
                                }
                                if (nu == su) { nxt[v] = sv; prv[sv] = v; }
                                else
                                {
                                    nxt[v] = nu; prv[nu] = v;
                                    nxt[last_u] = sv; prv[sv] = last_u;
                                    for (int a = nu; a != sv; a = nxt[a]) rof[a] = rv;
                                }
                                load[ru] = pre_u + suf_v; load[rv] = pre_v + suf_u;
                                moved = true;
                            }
                        }
                    }

                    if (moved)
                    {
                        push(u); push(v); push(pu); push(nu); push(pv); push(nv);
                    }
                }
            }

            // Rewrite routes (drop empty ones) and compute cost
            distance_t total = 0;
            std::vector <std::vector <int>> out;
            out.reserve(R);
            for (int r = 0; r < R; r++)
            {
                const int s = n + r;
                if (nxt[s] == s) continue;
                std::vector <int> route;
                int prev = s;
                for (int a = nxt[s]; a != s; a = nxt[a]) { total += D(prev, a); route.push_back(a); prev = a; }
                total += D(prev, s);
                out.push_back(std::move(route));
            }
            routes.swap(out);
            return total;
        }

        // Intra-route polish of local-id routes, returns total cost
        distance_t intra_route_polish(const Bucket_Workspace& ws, std::vector <std::vector <int>>& routes)
        {
            distance_t total = 0;
            std::vector <int> r;
            for (auto& route : routes)
            {
                r.clear();
                r.push_back(0);
                r.insert(r.end(), route.begin(), route.end());
                r.push_back(0);
                path_local_search(ws.x.data(), ws.y.data(), r);
                for (size_t i = 0; i < route.size(); i++) route[i] = r[i + 1];
                int prev = 0;
                for (int a : route) { total += ws.dist(prev, a); prev = a; }
                total += ws.d0[prev];
            }
            return total;
        }
    }

    void Solver::tsp_2OPT(
        const CVRP&             cvrp,
        std::vector <node_t>&   cities,
        std::vector <node_t>&   tour,
        int                     ncities) const
    {
        /*
        * tsp_2OPT: Intra-route local search (2-opt + Or-opt) on the open path
        * depot -> cities -> depot. `tour` is unused but kept for interface compatibility.
        */
        (void)tour;
        const int m = ncities;
        if (m < 2) return;

        std::vector <cord_t> px(m + 1), py(m + 1);
        px[0] = cvrp[cvrp.depot()].x; py[0] = cvrp[cvrp.depot()].y;
        std::vector <int> r(m + 2);
        r[0] = 0; r[m + 1] = 0;
        for (int i = 0; i < m; i++)
        {
            px[i + 1] = cvrp[cities[i]].x;
            py[i + 1] = cvrp[cities[i]].y;
            r[i + 1] = i + 1;
        }
        path_local_search(px.data(), py.data(), r);

        std::vector <node_t> out(m);
        for (int i = 0; i < m; i++) out[i] = cities[r[i + 1] - 1];
        cities.swap(out);
    }

    std::vector<std::vector<node_t>> Solver::process_2OPT(
        const CVRP&                             cvrp,
        const std::vector<std::vector<node_t>>& routes) const
    {
        std::vector<std::vector<node_t>> processed_routes;
        int nroutes = routes.size();
        processed_routes.reserve(nroutes);

        for (int i = 0; i < nroutes; ++i)
        {
            int sz = routes[i].size();
            std::vector<node_t> cities = routes[i];
            std::vector<node_t> tour;
            if (sz > 2)
            {
                tsp_2OPT(cvrp, cities, tour, sz);
            }
            processed_routes.push_back(std::move(cities));
        }
        return processed_routes;
    }

    void Solver::process_routes(
        const CVRP& cvrp,
        std::vector<std::vector<node_t>>& routes,
        distance_t& total_cost) const
    {
        /*
        * process_routes: For each route keep the cheapest of {as is, local search on it,
        * local search on its nearest neighbour reordering}.
        */
        auto processed_routes1  = process_tsp_approx(cvrp, routes);
        processed_routes1       = process_2OPT(cvrp, processed_routes1);
        auto processed_routes2  = process_2OPT(cvrp, routes);

        double min_cost = 0;
        for (size_t i = 0; i < routes.size(); i++)
        {
            auto route_cost     = get_route_distance(cvrp, routes[i]);
            auto route_cost1    = get_route_distance(cvrp, processed_routes1[i]);
            auto route_cost2    = get_route_distance(cvrp, processed_routes2[i]);

            if (std::min(route_cost1, route_cost2) >= route_cost)
            {
                min_cost        += route_cost;
            }
            else if (std::min(route_cost1, route_cost) >= route_cost2)
            {
                routes[i]       = std::move(processed_routes2[i]);
                min_cost        += route_cost2;
            }
            else
            {
                routes[i]       = std::move(processed_routes1[i]);
                min_cost        += route_cost1;
            }
        }
        total_cost = min_cost;
    }

    void Solver::solve_bucket(
        const CVRP&                         cvrp,
        const std::vector <node_t>&         bucket,
        const int                           bucket_id,
        const bool                          inner_parallel,
        std::vector <std::vector <node_t>>& routes,
        distance_t&                         cost) const
    {
        /*
        * solve_bucket: MST -> rho seeded random DFS orderings (cost only) -> optimal split
        * of the best few orderings -> intra-route local search on the winner.
        */
        routes.clear();
        cost = 0;
        if (bucket.size() <= 1) return;
        for (node_t u : bucket)
        {
            if (cvrp[u].demand > cvrp.capacity())
            {
                HANDLE_ERROR("Customer " + std::to_string(u) + " has demand larger than the vehicle capacity", true);
            }
        }

        Bucket_Workspace ws;
        ws.init(cvrp, bucket);
        double t0 = omp_get_wtime();

        // Neighbour lists serve both the sparse MST (large buckets) and the inter-route search
        std::vector <int> knn;
        build_knn(ws, KNN, knn);
        add_phase(P_KNN, omp_get_wtime() - t0); t0 = omp_get_wtime();

        if (ws.n > SPARSE_MST_MIN_NODES) sparse_mst(ws, knn, KNN);
        else                             construct_mst(ws);
        add_phase(P_MST, omp_get_wtime() - t0); t0 = omp_get_wtime();

        // Exploitation: rho random DFS orderings, each fully determined by its seed
        std::vector <distance_t> greedy_cost(rho);
        if (inner_parallel)
        {
            const int nth = omp_get_max_threads();
            std::vector <Bucket_Workspace> ws_pool(nth, ws);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < rho; t++)
            {
                greedy_cost[t] = random_dfs<false>(ws_pool[omp_get_thread_num()], mix_seed(seed, bucket_id, t));
            }
        }
        else
        {
            for (int t = 0; t < rho; t++)
            {
                greedy_cost[t] = random_dfs<false>(ws, mix_seed(seed, bucket_id, t));
            }
        }

        add_phase(P_DFS, omp_get_wtime() - t0); t0 = omp_get_wtime();

        // Pick the best few orderings; for each: optimal split -> intra LS -> inter-route LS -> intra LS
        const int K = std::min(rho, SPLIT_CANDIDATES);
        std::vector <int> order(rho);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
            [&](int a, int b) { return greedy_cost[a] < greedy_cost[b]; });

        // Stage 1: optimal split + intra-route polish for each of the K orderings
        struct Candidate { distance_t cost; std::vector <std::vector <int>> routes; };
        std::vector <Candidate> stage(K);
        for (int c = 0; c < K; c++)
        {
            const int t = order[c];
            random_dfs<true>(ws, mix_seed(seed, bucket_id, t));
            optimal_split(ws, stage[c].routes);
            add_phase(P_SPLIT, omp_get_wtime() - t0); t0 = omp_get_wtime();
            stage[c].cost = intra_route_polish(ws, stage[c].routes);
            add_phase(P_INTRA, omp_get_wtime() - t0); t0 = omp_get_wtime();
        }

        // Stage 2: inter-route search only on the M best of those
        const int M = std::min(K, INTER_CANDIDATES);
        std::partial_sort(stage.begin(), stage.begin() + M, stage.end(),
            [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

        distance_t best_cost = DBL_MAX;
        std::vector <std::vector <int>> best_local;
        for (int c = 0; c < M; c++)
        {
            inter_route_search(ws, knn, KNN, stage[c].routes);
            add_phase(P_INTER, omp_get_wtime() - t0); t0 = omp_get_wtime();
            const distance_t sc = intra_route_polish(ws, stage[c].routes);
            add_phase(P_INTRA, omp_get_wtime() - t0); t0 = omp_get_wtime();
            if (sc < best_cost)
            {
                best_cost = sc;
                best_local.swap(stage[c].routes);
            }
        }

        // Back to global ids, then the original per-route polish (also tries NN reordering)
        routes.resize(best_local.size());
        for (size_t r = 0; r < best_local.size(); r++)
        {
            routes[r].resize(best_local[r].size());
            for (size_t i = 0; i < best_local[r].size(); i++) routes[r][i] = bucket[best_local[r][i]];
        }
        cost = best_cost;
        process_routes(cvrp, routes, cost);
        add_phase(P_FINAL, omp_get_wtime() - t0);
    }

    Solver::Solver(
        const double _alpha,
        const int _rho,
        const std::uint64_t _seed)
        : alpha(_alpha), rho(_rho), seed(_seed ? _seed : std::random_device{}())
    {
    }

    Solution Solver::solve(
        const CVRP& cvrp) const
    {
        double maxMB_before_execution = get_curr_rss_mb();
        auto start = std::chrono::high_resolution_clock::now();

        distance_t final_cost   = 0.0;
        std::vector <std::vector<int>> final_routes;

        const int num_buckets = std::ceil(360.00 / alpha);
        std::vector <std::vector<node_t>> buckets(num_buckets);
        create_buckets(cvrp, buckets);

        // Two-level parallelism without oversubscription: when there are enough buckets
        // to fill the machine, run buckets in parallel and each bucket's rho loop serially.
        // Otherwise run buckets one at a time and parallelise the rho loop inside.
        const int nth = omp_get_max_threads();
        const bool buckets_parallel = num_buckets >= nth;

        // Largest buckets first so the dynamic schedule packs well
        std::vector <int> order(num_buckets);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
            [&](int a, int b) { return buckets[a].size() > buckets[b].size(); });

        #pragma omp parallel for schedule(dynamic, 1) if(buckets_parallel)
        for (int oi = 0; oi < num_buckets; oi++)
        {
            const int bucket_id = order[oi];
            std::vector <std::vector <node_t>> routes;
            distance_t cost = 0;
            solve_bucket(cvrp, buckets[bucket_id], bucket_id, !buckets_parallel, routes, cost);

            if (!routes.empty())
            {
                #pragma omp critical
                {
                    for (auto& route : routes)
                    {
                        final_routes.push_back(std::move(route));
                    }
                    final_cost += cost;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double maxMB_after_execution = get_peak_rss_mb();

        if (std::getenv("BPMDS_PROFILE"))
        {
            std::cerr << "phase CPU seconds (summed over threads):" << std::endl;
            for (int p = 0; p < P_COUNT; p++)
                std::cerr << "  " << PHASE_NAME[p] << ": " << phase_time[p] << std::endl;
        }

        double execution_time = std::chrono::duration<double>(end - start).count();
        double maxMB_difference = maxMB_after_execution - maxMB_before_execution;

        return Solution(execution_time, maxMB_difference, final_cost, final_routes);
    }
}
