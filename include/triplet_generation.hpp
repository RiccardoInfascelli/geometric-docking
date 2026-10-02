#ifndef TRIPLET_GENERATION_HPP
#define TRIPLET_GENERATION_HPP

#include <interaction/interaction_area.hpp>

#include <array>
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
    Chelated,
    Unknown
};

enum class AreaRole
{
    Ligand,
    Protein
};

struct Triplet
{
    std::array<std::size_t, 3> indices;
    std::array<InteractionKey, 3> keys;
    std::array<double, 3> distances;
};

std::vector<Triplet> generate_triplets(
    const std::vector<interaction::interaction_area>& areas,
    double max_distance,
    AreaRole role
);

#endif