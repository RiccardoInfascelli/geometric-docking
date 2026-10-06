#ifndef INTERACTION_SCORING_HPP
#define INTERACTION_SCORING_HPP

#include "matching_areas.hpp"
#include "protein_grid.hpp"

#include <cstddef>
#include <vector>

#include <interaction/interaction_area.hpp>

struct FastScoreWeights
{
    double hydrophobic = 1.0;
    double hydrogen_bond = 2.0;
    double halogen_bond = 2.0;

    // Conservative because the logical key does not always
    // distinguish ionic from cation-pi at grid level.
    double cation = 2.5;
    double anion = 3.0;
    double aromatic = 1.75;

    double metal = 3.0;
    double chelated = 3.0;
};

struct ExactScoreWeights
{
    double hydrophobic = 1.0;
    double pi_stacking = 1.5;
    double hydrogen_bond = 2.0;
    double halogen_bond = 2.0;
    double cation_pi = 2.5;
    double ionic = 3.0;
    double metal = 3.0;
};

struct FastInteractionScore
{
    std::size_t total_areas = 0;
    std::size_t areas_in_grid = 0;
    std::size_t areas_on_occupied_voxel = 0;
    std::size_t compatible_areas = 0;

    double weighted_score = 0.0;
};

struct ExactInteractionScore
{
    std::size_t total_interactions = 0;

    std::size_t satisfied_ligand_areas = 0;
    std::size_t satisfied_protein_areas = 0;

    std::size_t hydrophobic = 0;
    std::size_t hydrogen_bonds = 0;
    std::size_t halogen_bonds = 0;
    std::size_t ionic = 0;
    std::size_t cation_pi = 0;
    std::size_t pi_stacking = 0;
    std::size_t metal = 0;

    // Sum of the best exact interaction contribution for each
    // ligand interaction area. This avoids rewarding repeated
    // interactions on the same ligand feature multiple times.
    double weighted_score = 0.0;
};

double fast_interaction_weight(
    InteractionKey key,
    const FastScoreWeights& weights = {}
);

FastInteractionScore score_matching_areas_weighted(
    const std::vector<MatchingArea>& ligand_areas,
    const ProteinGrid& grid,
    const FastScoreWeights& weights = {}
);

ExactInteractionScore score_exact_interactions(
    const std::vector<interaction::interaction_area>& protein_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const ExactScoreWeights& weights = {}
);

#endif
