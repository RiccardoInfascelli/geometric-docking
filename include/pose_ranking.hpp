#ifndef POSE_RANKING_HPP
#define POSE_RANKING_HPP

#include "matching_areas.hpp"
#include "pose_generation.hpp"
#include "protein_grid.hpp"
#include <interaction/interaction_area.hpp>

#include <cstddef>
#include <limits>
#include <vector>


struct PoseRankingResult
{
    std::size_t input_pose_count{};

    std::vector<RankedPoseDiagnostic>
        unweighted_ranking;

    std::vector<RankedPoseDiagnostic>
        fast_weighted_ranking;

    std::vector<RankedPoseDiagnostic>
        exact_candidates;

    std::vector<RankedPoseDiagnostic>
        exact_count_ranking;

    std::vector<RankedPoseDiagnostic>
        exact_coverage_ranking;

    std::vector<RankedPoseDiagnostic>
        exact_weighted_ranking;

    double fast_scoring_seconds{};
    double exact_scoring_seconds{};

    std::size_t total_exact_interactions{};

    double best_prefilter_rmsd =
        std::numeric_limits<double>::infinity();

    std::vector<RankedPoseDiagnostic>
        exact_top50_weighted_ranking;

    double exact_top50_scoring_seconds{};
};


PoseRankingResult rank_poses(
    const std::vector<RankedPoseDiagnostic>& poses,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas,
    const ProteinGrid& protein_grid,
    std::size_t fast_top_n
);


PoseRankingResult rank_poses(
    const std::vector<RankedPoseDiagnostic>& poses,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas,
    const ProteinGrid& protein_grid,
    std::size_t fast_top_n);

struct ExactRankingResult
{
    std::vector<RankedPoseDiagnostic>
        scored_candidates;

    std::vector<RankedPoseDiagnostic>
        weighted_ranking;

    std::size_t total_exact_interactions{};

    double scoring_seconds{};
};


ExactRankingResult exact_score_and_rank(
    const std::vector<RankedPoseDiagnostic>& candidates,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas
);

#endif