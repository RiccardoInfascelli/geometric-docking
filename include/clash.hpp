#ifndef CLASH_HPP
#define CLASH_HPP

#include "geometry.hpp"
#include "protein_grid.hpp"

#include <GraphMol/ROMol.h>

#include <cstddef>
#include <vector>

struct ClashAtom
{
    double x{};
    double y{};
    double z{};
    double vdw_radius{};
};

struct ClashGrid
{
    double cell_size{};

    double xmin{};
    double ymin{};
    double zmin{};

    int nx{};
    int ny{};
    int nz{};

    double max_protein_radius{};

    std::vector<std::vector<ClashAtom>> cells;

    int index(int x, int y, int z) const;
    bool in_bounds(int x, int y, int z) const;
    int to_grid(double coordinate, double minimum) const;
};

ClashGrid build_clash_grid(
    const RDKit::ROMol& protein,
    double cell_size
);

double clash_score(
    const RDKit::ROMol& ligand,
    const ClashGrid& protein_grid,
    double tolerance
);

double clash_score_per_ligand_atom(
    const RDKit::ROMol& ligand,
    const ClashGrid& protein_grid,
    double tolerance
);

struct FastClashScore
{
    std::size_t total_atoms{};
    std::size_t occupied_atoms{};
    double fraction{};
};

FastClashScore score_fast_grid_clash(
    const RDKit::ROMol& ligand,
    const RigidTransform& transform,
    const ProteinGrid& grid
);

#endif
