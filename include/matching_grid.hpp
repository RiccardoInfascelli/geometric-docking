#ifndef MATCHING_GRID_HPP
#define MATCHING_GRID_HPP

#include "matching_areas.hpp"

#include <cstddef>
#include <vector>

struct MatchingGrid
{
    double cell_size{};

    double xmin{};
    double ymin{};
    double zmin{};

    int nx{};
    int ny{};
    int nz{};

    // CSR-like representation:
    // elementi della cella c =
    // area_indices[cell_offsets[c] ... cell_offsets[c + 1])
    std::vector<std::size_t> cell_offsets;
    std::vector<std::size_t> area_indices;

    int index(int x, int y, int z) const;

    bool in_bounds(int x, int y, int z) const;

    int to_grid(
        double coordinate,
        double minimum
    ) const;
};

MatchingGrid build_matching_grid(
    const std::vector<MatchingArea>& areas,
    double cell_size
);

#endif