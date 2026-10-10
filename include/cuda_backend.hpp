#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP

#include "cuda_triplet_types.hpp"
#include "cuda_grid.hpp"
#include "cuda_fast_scoring.hpp"

#include <cstdint>
#include <vector>



inline constexpr std::uint32_t
GPU_TRIPLET_KEY_NOT_FOUND = 0xFFFFFFFFu;


bool cuda_smoke_test();


bool cuda_upload_triplet_index(
    const GpuFlatTripletIndexHost& host_index,
    GpuFlatTripletIndexDevice& device_index
);


void cuda_free_triplet_index(
    GpuFlatTripletIndexDevice& device_index
);


bool cuda_lookup_triplet_keys_exact(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& key_indices
);

bool cuda_lookup_triplet_key_neighbors(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& neighbor_key_indices
);

bool cuda_count_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& record_counts
);

bool cuda_exclusive_prefix_sum(
    const std::vector<std::uint32_t>& counts,
    std::vector<std::uint32_t>& offsets
);

bool cuda_expand_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const std::vector<std::uint32_t>& candidate_offsets,
    std::vector<GpuTripletCandidate>& candidates
);

bool cuda_count_alignment_candidates(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    std::vector<std::uint32_t>& alignment_counts
);


bool cuda_expand_alignment_candidates(
    const std::vector<GpuTripletCandidate>& candidates,
    const std::vector<GpuLigandTripletOrders>& ligand_orders,
    const std::vector<std::uint32_t>& alignment_offsets,
    std::vector<GpuAlignmentCandidate>& alignments
);

bool cuda_upload_points(
    const std::vector<GpuPoint3>& host_points,
    GpuPointArrayDevice& device_points
);

bool cuda_download_points(
    const GpuPointArrayDevice& device_points,
    std::vector<GpuPoint3>& host_points
);

void cuda_free_points(
    GpuPointArrayDevice& device_points
);

bool cuda_compute_kabsch_transforms(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    std::vector<GpuRigidTransform>& transforms
);

bool cuda_upload_protein_grid(
    const GpuProteinGridHost& host_grid,
    GpuProteinGridDevice& device_grid
);


void cuda_free_protein_grid(
    GpuProteinGridDevice& device_grid
);


bool cuda_score_grid_clash(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuClashScore>& scores
);

bool cuda_upload_fast_matching_areas(
    const std::vector<GpuFastMatchingArea>& host_areas,
    GpuFastMatchingAreaArrayDevice& device_areas
);


void cuda_free_fast_matching_areas(
    GpuFastMatchingAreaArrayDevice& device_areas
);


bool cuda_score_fast_interactions(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuFastScore>& scores
);

#endif