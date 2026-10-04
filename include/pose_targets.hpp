#ifndef POSE_TARGETS_HPP
#define POSE_TARGETS_HPP

#include "matching_areas.hpp"
#include "protein_grid.hpp"

#include <GraphMol/ROMol.h>

#include <cstddef>
#include <vector>

struct PoseTarget
{
    std::size_t source_area_index{};
    interaction::point3D position{};
    InteractionKey key{};
};

std::vector<PoseTarget> generate_pose_targets(
    const std::vector<MatchingArea>& protein_areas,
    const ProteinGrid& grid,
    const RDKit::ROMol& protein,
    const RDKit::ROMol& pocket,
    double pocket_region_radius = 6.5
);

std::vector<MatchingArea>
filter_matching_areas_near_pocket(
    const std::vector<MatchingArea>& areas,
    const RDKit::ROMol& pocket,
    double radius
);

std::vector<MatchingArea>
filter_matching_areas_in_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const std::vector<interaction::interaction_area>& protein_areas,
    const RDKit::ROMol& pocket
);

#endif