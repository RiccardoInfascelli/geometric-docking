#ifndef MATCHING_AREAS_HPP
#define MATCHING_AREAS_HPP

#include <interaction/interaction_area.hpp>

#include <cstddef>
#include <vector>

enum class InteractionKey
{
    Hydrophobic,
    HydrogenDonor,
    HydrogenAcceptor,
    HalogenDonor,
    HalogenAcceptor,
    Cation,
    Anion,
    Aromatic,
    Metal,
    Chelated
};

struct MatchingArea
{
    std::size_t source_area_index;
    interaction::point3D centroid;
    InteractionKey key;
};

std::vector<MatchingArea> build_ligand_matching_areas(
    const std::vector<interaction::interaction_area>& areas
);

std::vector<MatchingArea> build_protein_matching_areas(
    const std::vector<interaction::interaction_area>& areas
);

#endif