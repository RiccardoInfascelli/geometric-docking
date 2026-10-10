#include "cuda_backend.hpp"

#include <cstdint>
#include <iostream>
#include <vector>


namespace
{

GpuTripletKey make_key(
    std::uint8_t k0,
    std::uint8_t k1,
    std::uint8_t k2,
    std::uint32_t d0,
    std::uint32_t d1,
    std::uint32_t d2
)
{
    GpuTripletKey key{};

    key.keys[0] = k0;
    key.keys[1] = k1;
    key.keys[2] = k2;

    key.distance_bins[0] = d0;
    key.distance_bins[1] = d1;
    key.distance_bins[2] = d2;

    return key;
}

} // namespace


int main()
{
    GpuFlatTripletIndexHost host_index;


    // ========================================================
    // Test receptor index
    // ========================================================

    host_index.keys = {
        make_key(
            0, 1, 2,
            1, 2, 3
        ),

        make_key(
            0, 1, 2,
            4, 5, 6
        ),

        make_key(
            1, 2, 3,
            7, 8, 9
        )
    };


    /*
     * key 0 -> 2 records
     * key 1 -> 1 record
     * key 2 -> 1 record
     */
    host_index.offsets = {
        0,
        2,
        3,
        4
    };


    host_index.records = {
        {{0, 1, 2}},
        {{10, 11, 12}},

        {{3, 4, 5}},

        {{6, 7, 8}}
    };


    GpuFlatTripletIndexDevice device_index;


    // ========================================================
    // Upload
    // ========================================================

    if (
        !cuda_upload_triplet_index(
            host_index,
            device_index
        )
    ) {
        std::cerr
            << "Triplet index upload failed\n";

        return 1;
    }


    // ========================================================
    // Exact lookup
    // ========================================================

    const std::vector<GpuTripletKey>
        exact_queries = {

            host_index.keys[0],
            host_index.keys[2],

            make_key(
                0, 1, 2,
                100, 100, 100
            )
        };


    std::vector<std::uint32_t>
        exact_results;


    if (
        !cuda_lookup_triplet_keys_exact(
            device_index,
            exact_queries,
            exact_results
        )
    ) {
        std::cerr
            << "Exact CUDA lookup failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        exact_results.size() != 3 ||
        exact_results[0] != 0 ||
        exact_results[1] != 2 ||
        exact_results[2] !=
            GPU_TRIPLET_KEY_NOT_FOUND
    ) {
        std::cerr
            << "Unexpected exact lookup results\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA exact triplet lookup passed\n";


    // ========================================================
    // 27-neighbor lookup
    // ========================================================

    const std::vector<GpuTripletKey>
        neighbor_queries = {

            make_key(
                0, 1, 2,
                4, 5, 6
            )
        };


    std::vector<std::uint32_t>
        neighbor_results;


    if (
        !cuda_lookup_triplet_key_neighbors(
            device_index,
            neighbor_queries,
            neighbor_results
        )
    ) {
        std::cerr
            << "CUDA neighbor lookup failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        neighbor_results.size() != 27
    ) {
        std::cerr
            << "Unexpected neighbor result count\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    bool found_exact_key = false;


    for (const auto key_index :
         neighbor_results) {

        if (key_index == 1) {
            found_exact_key = true;
            break;
        }
    }


    if (!found_exact_key) {

        std::cerr
            << "Exact key missing from "
               "27-neighbor lookup\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA 27-neighbor triplet lookup passed\n";


    // ========================================================
    // Record counting
    // ========================================================

    const std::vector<GpuTripletKey>
        count_queries = {

            make_key(
                0, 1, 2,
                2, 3, 4
            ),

            make_key(
                0, 1, 2,
                4, 5, 6
            ),

            make_key(
                9, 9, 9,
                4, 5, 6
            )
        };


    std::vector<std::uint32_t>
        record_counts;


    if (
        !cuda_count_triplet_neighbor_records(
            device_index,
            count_queries,
            record_counts
        )
    ) {
        std::cerr
            << "CUDA neighbor record counting failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        record_counts.size() != 3 ||
        record_counts[0] != 2 ||
        record_counts[1] != 1 ||
        record_counts[2] != 0
    ) {
        std::cerr
            << "Unexpected neighbor record counts\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA neighbor record counting passed\n";


    // ========================================================
    // Prefix sum
    // ========================================================

    std::vector<std::uint32_t>
        candidate_offsets;


    if (
        !cuda_exclusive_prefix_sum(
            record_counts,
            candidate_offsets
        )
    ) {
        std::cerr
            << "CUDA prefix sum failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    const std::vector<std::uint32_t>
        expected_offsets = {
            0,
            2,
            3,
            3
        };


    if (
        candidate_offsets !=
        expected_offsets
    ) {
        std::cerr
            << "Unexpected CUDA prefix-sum result\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA prefix sum passed\n";


    // ========================================================
    // Candidate expansion
    // ========================================================

    std::vector<GpuTripletCandidate>
        candidates;


    if (
        !cuda_expand_triplet_neighbor_records(
            device_index,
            count_queries,
            candidate_offsets,
            candidates
        )
    ) {
        std::cerr
            << "CUDA candidate expansion failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (candidates.size() != 3) {

        std::cerr
            << "Unexpected candidate count: "
            << candidates.size()
            << '\n';

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        candidates[0].ligand_triplet_index != 0 ||
        candidates[0].protein_indices[0] != 0 ||
        candidates[0].protein_indices[1] != 1 ||
        candidates[0].protein_indices[2] != 2
    ) {
        std::cerr
            << "Unexpected candidate 0\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        candidates[1].ligand_triplet_index != 0 ||
        candidates[1].protein_indices[0] != 10 ||
        candidates[1].protein_indices[1] != 11 ||
        candidates[1].protein_indices[2] != 12
    ) {
        std::cerr
            << "Unexpected candidate 1\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (
        candidates[2].ligand_triplet_index != 1 ||
        candidates[2].protein_indices[0] != 3 ||
        candidates[2].protein_indices[1] != 4 ||
        candidates[2].protein_indices[2] != 5
    ) {
        std::cerr
            << "Unexpected candidate 2\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA candidate expansion passed\n";

    // ========================================================
    // Equivalent ligand orders
    // ========================================================

    std::vector<GpuLigandTripletOrders>
        ligand_orders(3);


    /*
    * Ligand triplet 0 has two equivalent orders.
    */
    ligand_orders[0].equivalent_order_count = 2;

    ligand_orders[0].equivalent_orders[0][0] = 100;
    ligand_orders[0].equivalent_orders[0][1] = 101;
    ligand_orders[0].equivalent_orders[0][2] = 102;

    ligand_orders[0].equivalent_orders[1][0] = 102;
    ligand_orders[0].equivalent_orders[1][1] = 101;
    ligand_orders[0].equivalent_orders[1][2] = 100;


    /*
    * Ligand triplet 1 has one equivalent order.
    */
    ligand_orders[1].equivalent_order_count = 1;

    ligand_orders[1].equivalent_orders[0][0] = 200;
    ligand_orders[1].equivalent_orders[0][1] = 201;
    ligand_orders[1].equivalent_orders[0][2] = 202;


    /*
    * Ligand triplet 2 has no receptor match,
    * but keep one valid order for completeness.
    */
    ligand_orders[2].equivalent_order_count = 1;

    ligand_orders[2].equivalent_orders[0][0] = 300;
    ligand_orders[2].equivalent_orders[0][1] = 301;
    ligand_orders[2].equivalent_orders[0][2] = 302;


    /*
    * candidates currently are:
    *
    * candidate 0 -> ligand triplet 0
    * candidate 1 -> ligand triplet 0
    * candidate 2 -> ligand triplet 1
    *
    * Therefore:
    *
    * counts = [2, 2, 1]
    */
    std::vector<std::uint32_t>
        alignment_counts;


    if (
        !cuda_count_alignment_candidates(
            candidates,
            ligand_orders,
            alignment_counts
        )
    ) {
        std::cerr
            << "CUDA alignment candidate counting failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    const std::vector<std::uint32_t>
        expected_alignment_counts = {
            2,
            2,
            1
        };


    if (
        alignment_counts !=
        expected_alignment_counts
    ) {
        std::cerr
            << "Unexpected alignment candidate counts\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::vector<std::uint32_t>
        alignment_offsets;


    if (
        !cuda_exclusive_prefix_sum(
            alignment_counts,
            alignment_offsets
        )
    ) {
        std::cerr
            << "CUDA alignment prefix sum failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    const std::vector<std::uint32_t>
        expected_alignment_offsets = {
            0,
            2,
            4,
            5
        };


    if (
        alignment_offsets !=
        expected_alignment_offsets
    ) {
        std::cerr
            << "Unexpected alignment offsets\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::vector<GpuAlignmentCandidate>
        alignments;


    if (
        !cuda_expand_alignment_candidates(
            candidates,
            ligand_orders,
            alignment_offsets,
            alignments
        )
    ) {
        std::cerr
            << "CUDA alignment expansion failed\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    if (alignments.size() != 5) {

        std::cerr
            << "Unexpected alignment count: "
            << alignments.size()
            << '\n';

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    /*
    * First receptor candidate,
    * first ligand order.
    */
    if (
        alignments[0].ligand_indices[0] != 100 ||
        alignments[0].ligand_indices[1] != 101 ||
        alignments[0].ligand_indices[2] != 102 ||
        alignments[0].protein_indices[0] != 0 ||
        alignments[0].protein_indices[1] != 1 ||
        alignments[0].protein_indices[2] != 2
    ) {
        std::cerr
            << "Unexpected alignment 0\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    /*
    * First receptor candidate,
    * second ligand order.
    */
    if (
        alignments[1].ligand_indices[0] != 102 ||
        alignments[1].ligand_indices[1] != 101 ||
        alignments[1].ligand_indices[2] != 100 ||
        alignments[1].protein_indices[0] != 0 ||
        alignments[1].protein_indices[1] != 1 ||
        alignments[1].protein_indices[2] != 2
    ) {
        std::cerr
            << "Unexpected alignment 1\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    /*
    * Third receptor candidate belongs to
    * ligand triplet 1.
    */
    if (
        alignments[4].ligand_indices[0] != 200 ||
        alignments[4].ligand_indices[1] != 201 ||
        alignments[4].ligand_indices[2] != 202 ||
        alignments[4].protein_indices[0] != 3 ||
        alignments[4].protein_indices[1] != 4 ||
        alignments[4].protein_indices[2] != 5
    ) {
        std::cerr
            << "Unexpected alignment 4\n";

        cuda_free_triplet_index(
            device_index
        );

        return 1;
    }


    std::cout
        << "CUDA equivalent-order expansion passed\n";


    // ========================================================
    // Cleanup
    // ========================================================

    cuda_free_triplet_index(
        device_index
    );


    std::cout
        << "All CUDA triplet lookup tests passed\n";


    return 0;
}