#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP

#include "cuda_triplet_types.hpp"
#include "cuda_grid.hpp"
#include "cuda_fast_scoring.hpp"

#include <cstdint>
#include <vector>


inline constexpr std::uint32_t
GPU_TRIPLET_KEY_NOT_FOUND = 0xFFFFFFFFu;


// ============================================================
// Device-resident intermediate arrays
//
// These buffers allow the pose pipeline to keep intermediate
// results on the GPU instead of downloading them after every
// stage.
// ============================================================

struct GpuRigidTransformArrayDevice
{
    GpuRigidTransform* transforms{nullptr};
    std::uint32_t count{};
};


struct GpuClashScoreArrayDevice
{
    GpuClashScore* scores{nullptr};
    std::uint32_t count{};
};

struct GpuClashSurvivorArrayDevice
{
    GpuRigidTransform* transforms{nullptr};
    GpuClashScore* clash_scores{nullptr};
    std::uint32_t* alignment_indices{nullptr};
    std::uint32_t count{};
};


struct GpuFastScoreArrayDevice
{
    GpuFastScore* scores{nullptr};
    std::uint32_t count{};
};


// ============================================================
// General
// ============================================================

bool cuda_smoke_test();


// ============================================================
// Triplet index
// ============================================================

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


// ============================================================
// Point arrays
// ============================================================

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


// ============================================================
// Kabsch
//
// Legacy API:
//     GPU computation -> host vector.
//
// Device API:
//     GPU computation -> GPU buffer.
// ============================================================

bool cuda_compute_kabsch_transforms(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    std::vector<GpuRigidTransform>& transforms
);


bool cuda_compute_kabsch_transforms_device(
    const GpuPointArrayDevice& ligand_points,
    const GpuPointArrayDevice& protein_points,
    const std::vector<GpuAlignmentCandidate>& alignments,
    GpuRigidTransformArrayDevice& transforms
);


bool cuda_download_rigid_transforms(
    const GpuRigidTransformArrayDevice& device_transforms,
    std::vector<GpuRigidTransform>& host_transforms
);


void cuda_free_rigid_transforms(
    GpuRigidTransformArrayDevice& device_transforms
);


// ============================================================
// Protein grid
// ============================================================

bool cuda_upload_protein_grid(
    const GpuProteinGridHost& host_grid,
    GpuProteinGridDevice& device_grid
);


void cuda_free_protein_grid(
    GpuProteinGridDevice& device_grid
);


// ============================================================
// Grid clash
//
// Legacy API:
//     host transforms -> GPU -> host scores.
//
// Device API:
//     GPU transforms -> GPU scores.
// ============================================================

bool cuda_score_grid_clash(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuClashScore>& scores
);


bool cuda_score_grid_clash_device(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransformArrayDevice& transforms,
    GpuClashScoreArrayDevice& scores
);


bool cuda_download_clash_scores(
    const GpuClashScoreArrayDevice& device_scores,
    std::vector<GpuClashScore>& host_scores
);


void cuda_free_clash_scores(
    GpuClashScoreArrayDevice& device_scores
);


bool cuda_compact_clash_survivors(
    const GpuRigidTransformArrayDevice& transforms,
    const GpuClashScoreArrayDevice& clash_scores,
    double clash_threshold,
    GpuClashSurvivorArrayDevice& survivors,
    std::uint32_t& valid_transform_count
);


bool cuda_download_clash_survivors(
    const GpuClashSurvivorArrayDevice& device_survivors,
    std::vector<GpuRigidTransform>& host_transforms,
    std::vector<GpuClashScore>& host_clash_scores,
    std::vector<std::uint32_t>& host_alignment_indices
);


void cuda_free_clash_survivors(
    GpuClashSurvivorArrayDevice& survivors
);

// ============================================================
// Fast matching areas
// ============================================================

bool cuda_upload_fast_matching_areas(
    const std::vector<GpuFastMatchingArea>& host_areas,
    GpuFastMatchingAreaArrayDevice& device_areas
);


void cuda_free_fast_matching_areas(
    GpuFastMatchingAreaArrayDevice& device_areas
);


// ============================================================
// Fast interaction scoring
//
// We keep the current host-vector API for now.
// A device-resident scoring path will be connected after
// GPU clash filtering/compaction is implemented.
// ============================================================

bool cuda_score_fast_interactions(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuFastScore>& scores
);


#endif