#ifndef BUCKET_PARTITIONED_MDS_H
#define BUCKET_PARTITIONED_MDS_H

#include "Utils.h"
#include <vector>
#include <string>
#include <cfloat>
#include <cstdint>
#include <iostream>

namespace Bucket_Partitioned_MDS
{
    class CVRP
    {
        /*
        * CVRP: Class for maintaining CVRP
        */

    private:
        capacity_t _capacity;
        int _size;
        std::vector <Point> node;
        std::string type;
        const node_t _depot = 0; // depot is always 0

    public:
        CVRP(
            std::istream&);
        distance_t distance(
            const node_t,
            const node_t) const;
        const Point& operator[](
            const node_t) const;
        void print(
            std::ostream&) const;
        // Getters
        capacity_t capacity() const;
        int size() const;
        node_t depot() const;
    };

    class Solution
    {
    private:
        double time_for_solving;
        double maxMB_difference;
        double cost;
        std::vector <std::vector <node_t>> routes;

        distance_t get_total_cost_of_routes(
            const CVRP& cvrp);
    public:
        Solution(
            const double,
            const double,
            const double,
            const std::vector <std::vector <node_t>>&);
        bool verify(
            const CVRP&) const;
        void print(
            std::ostream&) const;
    };

    // Per-bucket scratch state. Declared here so Solver can reuse it; defined in Solver.cpp.
    struct Bucket_Workspace;

    class Solver
    {
        /*
        * Solver: Bucket Partitioned MDS solver for solving CVRP
        */

    private:
        const double alpha;
        const int rho;
        const std::uint64_t seed;

        void construct_mst(
            Bucket_Workspace&) const;

        distance_t get_route_distance(
            const CVRP&,
            const std::vector <node_t>&) const;

        void tsp_approx(
            const CVRP&,
            std::vector <node_t>&,
            std::vector <node_t>&,
            node_t) const;

        std::vector<std::vector<node_t>> process_tsp_approx(
            const CVRP&,
            const std::vector<std::vector<node_t>>&) const;

        void tsp_2OPT(
            const CVRP&,
            std::vector <node_t>&,
            std::vector <node_t>&,
            int) const;

        std::vector<std::vector<node_t>> process_2OPT(
            const CVRP&,
            const std::vector<std::vector<node_t>>&) const;

        void process_routes(
            const CVRP&,
            std::vector <std::vector<int>>&,
            distance_t&) const;

        void solve_bucket(
            const CVRP&,
            const std::vector <node_t>&,
            const int,
            const bool,
            std::vector <std::vector <node_t>>&,
            distance_t&) const;

    public:
        Solver(
            const double,
            const int,
            const std::uint64_t = 0);

        Solution solve(
            const CVRP&) const;

        void create_buckets(
            const CVRP&,
            std::vector <std::vector <node_t>>&) const;
    };
}

#endif
