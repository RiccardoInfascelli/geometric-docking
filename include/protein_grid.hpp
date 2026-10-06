#ifndef PROTEIN_GRID_HPP
#define PROTEIN_GRID_HPP

#include "matching_areas.hpp"

#include <GraphMol/ROMol.h>

#include <cstddef>
#include <vector>


struct GridBounds
{
    double xmin{};
    double xmax{};

    double ymin{};
    double ymax{};

    double zmin{};
    double zmax{};
};


struct ProteinVoxel
{
    bool occupied = false;
    InteractionMask interactions = 0;
};


struct ProteinGrid
{
    double voxel_size{};

    double xmin{};
    double ymin{};
    double zmin{};

    int nx{};
    int ny{};
    int nz{};

    std::vector<ProteinVoxel> voxels;

    std::size_t index(
        int x,
        int y,
        int z
    ) const;

    bool in_bounds(
        int x,
        int y,
        int z
    ) const;

    int to_grid(
        double coordinate,
        double minimum
    ) const;
};


GridBounds compute_pocket_bounds(
    const RDKit::ROMol& pocket
);


ProteinGrid build_empty_protein_grid(
    const RDKit::ROMol& pocket,
    double voxel_size,
    double margin
);

void mark_protein_occupancy(
    const RDKit::ROMol& protein,
    ProteinGrid& grid
);

void mark_interaction_regions(
    const std::vector<MatchingArea>& areas,
    ProteinGrid& grid
);

struct GridPoseScore
{
    std::size_t total_areas = 0;
    std::size_t areas_in_grid = 0;
    std::size_t areas_on_occupied_voxel = 0;
    std::size_t compatible_areas = 0;
};

GridPoseScore score_ligand_on_grid(
    const std::vector<MatchingArea>& ligand_areas,
    const ProteinGrid& grid
);

std::size_t count_occupied_voxels(
    const ProteinGrid& grid
);

std::size_t count_interaction_voxels(
    const ProteinGrid& grid
);

std::size_t count_multi_interaction_voxels(
    const ProteinGrid& grid
);


#endif // PROTEIN_GRID_HPP