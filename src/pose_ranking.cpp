#include "pose_ranking.hpp"

#include "geometry.hpp"
#include "interaction_scoring.hpp"

#include <algorithm>
#include <chrono>


PoseRankingResult rank_poses(
    const std::vector<RankedPoseDiagnostic>& surviving_poses,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& pocket_protein_areas,
    const ProteinGrid& protein_grid,
    std::size_t fast_prefilter_top_n)
{
    PoseRankingResult result;

    result.input_pose_count =
        surviving_poses.size();


    // ========================================================
    // FAST WEIGHTED SCORING
    //
    // Cheap grid-based score on every clash survivor.
    // ========================================================

    auto scored_poses =
        surviving_poses;


    const auto fast_start =
        std::chrono::steady_clock::now();


    for (auto& pose : scored_poses) {

        const auto transformed_matching =
            transform_matching_areas(
                ligand_matching_areas,
                pose.transform
            );


        const auto fast_score =
            score_matching_areas_weighted(
                transformed_matching,
                protein_grid
            );


        pose.fast_weighted_score =
            fast_score.weighted_score;
    }


    const auto fast_end =
        std::chrono::steady_clock::now();


    result.fast_scoring_seconds =
        std::chrono::duration<double>(
            fast_end -
            fast_start
        ).count();


    // ========================================================
    // UNWEIGHTED GRID RANKING
    //
    // Diagnostic baseline.
    // RMSD NEVER participates in sorting.
    // ========================================================

    result.unweighted_ranking =
        scored_poses;


    std::sort(
        result.unweighted_ranking.begin(),
        result.unweighted_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.compatible_areas !=
                b.compatible_areas
            ) {
                return
                    a.compatible_areas >
                    b.compatible_areas;
            }

            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // ========================================================
    // FAST WEIGHTED RANKING
    //
    // This is the FAST-ONLY experiment.
    // ========================================================

    result.fast_weighted_ranking =
        scored_poses;


    std::sort(
        result.fast_weighted_ranking.begin(),
        result.fast_weighted_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.fast_weighted_score !=
                b.fast_weighted_score
            ) {
                return
                    a.fast_weighted_score >
                    b.fast_weighted_score;
            }


            if (
                a.compatible_areas !=
                b.compatible_areas
            ) {
                return
                    a.compatible_areas >
                    b.compatible_areas;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // ========================================================
    // FAST PREFILTER
    //
    // Keep only the best N fast poses for exact chemistry.
    // Normally N = 500.
    // ========================================================

    result.exact_candidates =
        result.fast_weighted_ranking;


    if (
        result.exact_candidates.size() >
        fast_prefilter_top_n
    ) {
        result.exact_candidates.resize(
            fast_prefilter_top_n
        );
    }


    // --------------------------------------------------------
    // Oracle diagnostic only:
    // best RMSD that survived the fast prefilter.
    // --------------------------------------------------------

    result.best_prefilter_rmsd =
        std::numeric_limits<double>::infinity();


    for (const auto& pose :
         result.exact_candidates) {

        result.best_prefilter_rmsd =
            std::min(
                result.best_prefilter_rmsd,
                pose.rmsd
            );
    }


    // ========================================================
    // EXACT SCORING
    //
    // We score the fast top-N ONCE.
    //
    // While doing so we record how much time was needed
    // to score only the first 50.
    // ========================================================

    const auto exact_start =
        std::chrono::steady_clock::now();


    result.total_exact_interactions = 0;


    const std::size_t top50_count =
        std::min<std::size_t>(
            50,
            result.exact_candidates.size()
        );


    for (
        std::size_t pose_index = 0;
        pose_index < result.exact_candidates.size();
        ++pose_index
    ) {
        auto& pose =
            result.exact_candidates[
                pose_index
            ];


        const auto transformed_areas =
            transform_interaction_areas(
                ligand_areas,
                pose.transform
            );


        pose.exact_score =
            score_exact_interactions(
                pocket_protein_areas,
                transformed_areas
            );


        result.total_exact_interactions +=
            pose.exact_score
                .total_interactions;


        // Time required to exact-score only
        // the first 50 fast-ranked poses.
        if (
            pose_index + 1 ==
            top50_count
        ) {
            const auto exact_top50_end =
                std::chrono::steady_clock::now();


            result.exact_top50_scoring_seconds =
                std::chrono::duration<double>(
                    exact_top50_end -
                    exact_start
                ).count();
        }
    }


    const auto exact_end =
        std::chrono::steady_clock::now();


    result.exact_scoring_seconds =
        std::chrono::duration<double>(
            exact_end -
            exact_start
        ).count();


    // No exact candidates.
    if (result.exact_candidates.empty()) {

        result.exact_top50_scoring_seconds =
            0.0;
    }


    // ========================================================
    // EXACT TOP-50 EXPERIMENT
    //
    // IMPORTANT:
    // no chemistry is recomputed here.
    //
    // These are simply the first 50 candidates from the
    // FAST ranking, whose exact scores have already been
    // computed above.
    // ========================================================

    result.exact_top50_weighted_ranking =
        result.exact_candidates;


    if (
        result.exact_top50_weighted_ranking.size() >
        50
    ) {
        result.exact_top50_weighted_ranking.resize(
            50
        );
    }


    // ========================================================
    // EXACT WEIGHTED COMPARATOR
    //
    // Shared by TOP-50 and TOP-N.
    //
    // RMSD is NEVER used.
    // ========================================================

    const auto exact_weighted_better =
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.exact_score.weighted_score !=
                b.exact_score.weighted_score
            ) {
                return
                    a.exact_score.weighted_score >
                    b.exact_score.weighted_score;
            }


            if (
                a.exact_score
                    .satisfied_ligand_areas !=
                b.exact_score
                    .satisfied_ligand_areas
            ) {
                return
                    a.exact_score
                        .satisfied_ligand_areas >
                    b.exact_score
                        .satisfied_ligand_areas;
            }


            if (
                a.exact_score.total_interactions !=
                b.exact_score.total_interactions
            ) {
                return
                    a.exact_score.total_interactions >
                    b.exact_score.total_interactions;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        };


    std::sort(
        result.exact_top50_weighted_ranking.begin(),
        result.exact_top50_weighted_ranking.end(),
        exact_weighted_better
    );


    // ========================================================
    // EXACT INTERACTION-COUNT RANKING
    //
    // Diagnostic only.
    // ========================================================

    result.exact_count_ranking =
        result.exact_candidates;


    std::sort(
        result.exact_count_ranking.begin(),
        result.exact_count_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.exact_score.total_interactions !=
                b.exact_score.total_interactions
            ) {
                return
                    a.exact_score.total_interactions >
                    b.exact_score.total_interactions;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // ========================================================
    // EXACT LIGAND-COVERAGE RANKING
    //
    // Diagnostic only.
    // ========================================================

    result.exact_coverage_ranking =
        result.exact_candidates;


    std::sort(
        result.exact_coverage_ranking.begin(),
        result.exact_coverage_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b)
        {
            if (
                a.exact_score
                    .satisfied_ligand_areas !=
                b.exact_score
                    .satisfied_ligand_areas
            ) {
                return
                    a.exact_score
                        .satisfied_ligand_areas >
                    b.exact_score
                        .satisfied_ligand_areas;
            }


            if (
                a.exact_score.total_interactions !=
                b.exact_score.total_interactions
            ) {
                return
                    a.exact_score.total_interactions >
                    b.exact_score.total_interactions;
            }


            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    // ========================================================
    // EXACT WEIGHTED TOP-N
    //
    // Normally:
    // fast TOP 500 -> exact weighted reranking.
    // ========================================================

    result.exact_weighted_ranking =
        result.exact_candidates;


    std::sort(
        result.exact_weighted_ranking.begin(),
        result.exact_weighted_ranking.end(),
        exact_weighted_better
    );


    return result;
}


ExactRankingResult exact_score_and_rank(
    const std::vector<RankedPoseDiagnostic>& candidates,
    const std::vector<interaction::interaction_area>& ligand_areas,
    const std::vector<interaction::interaction_area>& protein_areas)
{
    ExactRankingResult result;

    result.scored_candidates =
        candidates;


    // ========================================================
    // Exact scoring
    // ========================================================

    const auto start =
        std::chrono::steady_clock::now();


    for (auto& pose :
         result.scored_candidates) {

        const auto transformed_areas =
            transform_interaction_areas(
                ligand_areas,
                pose.transform
            );

        pose.exact_score =
            score_exact_interactions(
                protein_areas,
                transformed_areas
            );


        result.total_exact_interactions +=
            pose.exact_score.total_interactions;
    }


    const auto end =
        std::chrono::steady_clock::now();


    result.scoring_seconds =
        std::chrono::duration<double>(
            end - start
        ).count();


    // ========================================================
    // Exact weighted ranking
    // ========================================================

    result.weighted_ranking =
        result.scored_candidates;


    std::sort(
        result.weighted_ranking.begin(),
        result.weighted_ranking.end(),
        [](
            const RankedPoseDiagnostic& a,
            const RankedPoseDiagnostic& b
        )
        {
            if (
                a.exact_score.weighted_score !=
                b.exact_score.weighted_score
            ) {
                return
                    a.exact_score.weighted_score >
                    b.exact_score.weighted_score;
            }

            if (
                a.exact_score.satisfied_ligand_areas !=
                b.exact_score.satisfied_ligand_areas
            ) {
                return
                    a.exact_score.satisfied_ligand_areas >
                    b.exact_score.satisfied_ligand_areas;
            }

            if (
                a.exact_score.total_interactions !=
                b.exact_score.total_interactions
            ) {
                return
                    a.exact_score.total_interactions >
                    b.exact_score.total_interactions;
            }

            return
                a.occupied_atoms <
                b.occupied_atoms;
        }
    );


    return result;
}