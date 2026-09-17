#!/usr/bin/env python3
"""Independent checker: recompute a .sol against its .vrp without using solver code.

usage: python3 Scripts/verify_solution.py instance.vrp solution.sol
Checks: every customer served exactly once, every route within capacity,
and the recomputed Euclidean cost against the cost line in the .sol.
"""
import math
import sys


def read_vrp(path):
    coords, demands, cap, dim = {}, {}, None, None
    section = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0].startswith("DIMENSION"):
                dim = int(t[-1])
            elif t[0].startswith("CAPACITY"):
                cap = float(t[-1])
            elif t[0].startswith("NODE_COORD_SECTION"):
                section = "coord"
            elif t[0].startswith("DEMAND_SECTION"):
                section = "demand"
            elif t[0].startswith("DEPOT_SECTION") or t[0] == "EOF":
                section = None
            elif section == "coord":
                coords[int(t[0]) - 1] = (float(t[1]), float(t[2]))
            elif section == "demand":
                demands[int(t[0]) - 1] = float(t[1])
    assert len(coords) == dim and len(demands) == dim, "vrp parse mismatch"
    return coords, demands, cap


def main():
    if len(sys.argv) != 3:
        print("usage: python3 Scripts/verify_solution.py instance.vrp solution.sol")
        sys.exit(2)
    vrp, sol = sys.argv[1], sys.argv[2]
    coords, demands, cap = read_vrp(vrp)
    n = len(coords)
    depot = coords[0]

    seen = [0] * n
    seen[0] = 1
    reported_cost = None
    total = 0.0
    nroutes = 0
    worst_load = 0.0
    with open(sol) as f:
        for line in f:
            if line.startswith("Cost:"):
                reported_cost = float(line.split()[1])
            elif line.startswith("Route #"):
                nodes = [int(x) for x in line.split(":", 1)[1].split()]
                nroutes += 1
                load = 0.0
                prev = depot
                for v in nodes:
                    seen[v] += 1
                    load += demands[v]
                    c = coords[v]
                    total += math.hypot(c[0] - prev[0], c[1] - prev[1])
                    prev = c
                total += math.hypot(depot[0] - prev[0], depot[1] - prev[1])
                worst_load = max(worst_load, load)
                if load > cap + 1e-9:
                    print(f"FAIL capacity: route {nroutes} load {load} > {cap}")
                    sys.exit(1)

    missing = [v for v in range(1, n) if seen[v] == 0]
    dup = [v for v in range(1, n) if seen[v] > 1]
    if missing or dup:
        print(f"FAIL coverage: {len(missing)} missing, {len(dup)} duplicated")
        sys.exit(1)
    if reported_cost is None:
        print("FAIL: no Cost line")
        sys.exit(1)
    abs_diff = abs(total - reported_cost)
    rel = abs_diff / reported_cost
    print(f"customers={n - 1} routes={nroutes} capacity={cap:g} max_route_load={worst_load:g}")
    print(f"recomputed={total:.4f} reported={reported_cost:.4f} rel_diff={rel:.2e}")
    # Solution::print writes the cost with fixed precision of four decimal
    # places.  Therefore an independent recomputation is expected to differ by
    # a small rounding amount even when the route is exactly correct.  Allow
    # 1e-4 absolute error for that printed precision, plus a tiny relative
    # allowance for a different order of floating-point additions at 1M scale.
    tolerance = max(1e-4, abs(reported_cost) * 1e-12)
    if abs_diff > tolerance:
        print(f"FAIL: cost mismatch (absolute difference {abs_diff:.6g} > tolerance {tolerance:.6g})")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
