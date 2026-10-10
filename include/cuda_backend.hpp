#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP

#include "cuda_triplet_types.hpp"

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

#endif