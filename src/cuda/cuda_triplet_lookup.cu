#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <cub/cub.cuh>


namespace
{

__device__ int compare_triplet_keys(
    const GpuTripletKey& a,
    const GpuTripletKey& b
)
{
    for (int i = 0; i < 3; ++i) {

        if (a.keys[i] < b.keys[i])
            return -1;

        if (a.keys[i] > b.keys[i])
            return 1;
    }


    for (int i = 0; i < 3; ++i) {

        if (a.distance_bins[i] < b.distance_bins[i])
            return -1;

        if (a.distance_bins[i] > b.distance_bins[i])
            return 1;
    }


    return 0;
}


__device__ std::uint32_t find_triplet_key(
    const GpuTripletKey* keys,
    std::uint32_t key_count,
    const GpuTripletKey& query
)
{
    std::uint32_t begin = 0;
    std::uint32_t end = key_count;


    while (begin < end) {

        const std::uint32_t middle =
            begin + (end - begin) / 2;


        const int comparison =
            compare_triplet_keys(
                keys[middle],
                query
            );


        if (comparison < 0) {
            begin = middle + 1;
        }
        else {
            end = middle;
        }
    }


    if (
        begin < key_count &&
        compare_triplet_keys(
            keys[begin],
            query
        ) == 0
    ) {
        return begin;
    }


    return 0xFFFFFFFFu;
}


__global__ void triplet_exact_lookup_kernel(
    const GpuTripletKey* index_keys,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    std::uint32_t* results
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (query_index >= query_count)
        return;


    results[query_index] =
        find_triplet_key(
            index_keys,
            index_key_count,
            queries[query_index]
        );
}

__global__ void triplet_neighbor_lookup_kernel(
    const GpuTripletKey* index_keys,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    std::uint32_t* results
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (query_index >= query_count)
        return;


    const GpuTripletKey base =
        queries[query_index];


    std::uint32_t slot = 0;


    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                GpuTripletKey candidate =
                    base;


                const int deltas[3] = {
                    d0,
                    d1,
                    d2
                };


                bool valid = true;


                for (int i = 0; i < 3; ++i) {

                    if (
                        deltas[i] < 0 &&
                        candidate.distance_bins[i] == 0
                    ) {
                        valid = false;
                        break;
                    }


                    candidate.distance_bins[i] =
                        static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(
                                candidate.distance_bins[i]
                            ) +
                            deltas[i]
                        );
                }


                const std::uint32_t output_index =
                    query_index * 27 + slot;


                if (valid) {
                    results[output_index] =
                        find_triplet_key(
                            index_keys,
                            index_key_count,
                            candidate
                        );
                }
                else {
                    results[output_index] =
                        GPU_TRIPLET_KEY_NOT_FOUND;
                }


                ++slot;
            }
        }
    }
}

__global__ void triplet_neighbor_count_kernel(
    const GpuTripletKey* index_keys,
    const std::uint32_t* index_offsets,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    std::uint32_t* counts
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (query_index >= query_count)
        return;


    const GpuTripletKey base =
        queries[query_index];

    std::uint32_t count = 0;


    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                GpuTripletKey candidate =
                    base;

                const int deltas[3] = {
                    d0,
                    d1,
                    d2
                };

                bool valid = true;


                for (int i = 0; i < 3; ++i) {

                    if (
                        deltas[i] < 0 &&
                        candidate.distance_bins[i] == 0
                    ) {
                        valid = false;
                        break;
                    }

                    candidate.distance_bins[i] =
                        static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(
                                candidate.distance_bins[i]
                            ) +
                            deltas[i]
                        );
                }


                if (!valid)
                    continue;


                const std::uint32_t key_index =
                    find_triplet_key(
                        index_keys,
                        index_key_count,
                        candidate
                    );


                if (
                    key_index ==
                    GPU_TRIPLET_KEY_NOT_FOUND
                ) {
                    continue;
                }


                const std::uint32_t begin =
                    index_offsets[key_index];

                const std::uint32_t end =
                    index_offsets[key_index + 1];


                count += end - begin;
            }
        }
    }


    counts[query_index] = count;
}

__global__ void write_final_prefix_offset_kernel(
    const std::uint32_t* counts,
    std::uint32_t count,
    std::uint32_t* offsets
)
{
    if (
        blockIdx.x == 0 &&
        threadIdx.x == 0
    ) {
        offsets[count] =
            offsets[count - 1] +
            counts[count - 1];
    }
}


__global__ void triplet_neighbor_expand_kernel(
    const GpuTripletKey* index_keys,
    const std::uint32_t* index_offsets,
    const GpuProteinTripletRecord* index_records,
    std::uint32_t index_key_count,
    const GpuTripletKey* queries,
    std::uint32_t query_count,
    const std::uint32_t* candidate_offsets,
    GpuTripletCandidate* candidates
)
{
    const std::uint32_t query_index =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (query_index >= query_count)
        return;


    const GpuTripletKey base =
        queries[query_index];


    std::uint32_t output_index =
        candidate_offsets[query_index];


    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {

                GpuTripletKey candidate_key =
                    base;


                const int deltas[3] = {
                    d0,
                    d1,
                    d2
                };


                bool valid = true;


                for (int i = 0; i < 3; ++i) {

                    if (
                        deltas[i] < 0 &&
                        candidate_key.distance_bins[i] == 0
                    ) {
                        valid = false;
                        break;
                    }


                    candidate_key.distance_bins[i] =
                        static_cast<std::uint32_t>(
                            static_cast<std::int64_t>(
                                candidate_key.distance_bins[i]
                            ) +
                            deltas[i]
                        );
                }


                if (!valid)
                    continue;


                const std::uint32_t key_index =
                    find_triplet_key(
                        index_keys,
                        index_key_count,
                        candidate_key
                    );


                if (
                    key_index ==
                    GPU_TRIPLET_KEY_NOT_FOUND
                ) {
                    continue;
                }


                const std::uint32_t begin =
                    index_offsets[key_index];

                const std::uint32_t end =
                    index_offsets[key_index + 1];


                for (
                    std::uint32_t record_index = begin;
                    record_index < end;
                    ++record_index
                ) {
                    const auto record =
                        index_records[record_index];


                    GpuTripletCandidate output{};

                    output.ligand_triplet_index =
                        query_index;

                    output.protein_indices[0] =
                        record.indices[0];

                    output.protein_indices[1] =
                        record.indices[1];

                    output.protein_indices[2] =
                        record.indices[2];


                    candidates[output_index] =
                        output;


                    ++output_index;
                }
            }
        }
    }
}

} // namespace


bool cuda_lookup_triplet_keys_exact(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& key_indices
)
{
    key_indices.clear();


    if (queries.empty())
        return true;


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    if (
        device_index.key_count > 0 &&
        device_index.keys == nullptr
    ) {
        return false;
    }


    GpuTripletKey* device_queries = nullptr;

    std::uint32_t* device_results =
        nullptr;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t result_bytes =
        queries.size() *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_results
            ),
            result_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    constexpr std::uint32_t block_size = 256;

    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );

    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_exact_lookup_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.key_count,
            device_queries,
            query_count,
            device_results
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    key_indices.resize(
        queries.size()
    );


    if (
        cudaMemcpy(
            key_indices.data(),
            device_results,
            result_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        key_indices.clear();

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_results);
    cudaFree(device_queries);


    return true;
}

bool cuda_lookup_triplet_key_neighbors(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& neighbor_key_indices
)
{
    neighbor_key_indices.clear();


    if (queries.empty())
        return true;


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    if (
        device_index.key_count > 0 &&
        device_index.keys == nullptr
    ) {
        return false;
    }


    GpuTripletKey* device_queries =
        nullptr;

    std::uint32_t* device_results =
        nullptr;


    const std::size_t result_count =
        queries.size() * 27;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t result_bytes =
        result_count *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_results
            ),
            result_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );


    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_neighbor_lookup_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.key_count,
            device_queries,
            query_count,
            device_results
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    neighbor_key_indices.resize(
        result_count
    );


    if (
        cudaMemcpy(
            neighbor_key_indices.data(),
            device_results,
            result_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        neighbor_key_indices.clear();

        cudaFree(device_results);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_results);
    cudaFree(device_queries);


    return true;
}

bool cuda_count_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    std::vector<std::uint32_t>& record_counts
)
{
    record_counts.clear();


    if (queries.empty())
        return true;


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    if (
        device_index.keys == nullptr ||
        device_index.offsets == nullptr
    ) {
        return false;
    }


    GpuTripletKey* device_queries =
        nullptr;

    std::uint32_t* device_counts =
        nullptr;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t count_bytes =
        queries.size() *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_counts
            ),
            count_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );


    constexpr std::uint32_t block_size =
        256;

    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_neighbor_count_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.offsets,
            device_index.key_count,
            device_queries,
            query_count,
            device_counts
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    record_counts.resize(
        queries.size()
    );


    if (
        cudaMemcpy(
            record_counts.data(),
            device_counts,
            count_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        record_counts.clear();

        cudaFree(device_counts);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_counts);
    cudaFree(device_queries);


    return true;
}

bool cuda_exclusive_prefix_sum(
    const std::vector<std::uint32_t>& counts,
    std::vector<std::uint32_t>& offsets
)
{
    offsets.clear();


    if (counts.empty()) {
        offsets.push_back(0);
        return true;
    }


    if (
        counts.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    std::uint64_t total = 0;

    for (const auto count : counts) {

        total += count;

        if (
            total >
            std::numeric_limits<std::uint32_t>::max()
        ) {
            return false;
        }
    }


    const std::uint32_t count =
        static_cast<std::uint32_t>(
            counts.size()
        );


    std::uint32_t* device_counts =
        nullptr;

    std::uint32_t* device_offsets =
        nullptr;

    void* temporary_storage =
        nullptr;


    const std::size_t count_bytes =
        counts.size() *
        sizeof(std::uint32_t);

    const std::size_t offset_bytes =
        (counts.size() + 1) *
        sizeof(std::uint32_t);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_counts
            ),
            count_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_offsets
            ),
            offset_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_counts);
        return false;
    }


    if (
        cudaMemcpy(
            device_counts,
            counts.data(),
            count_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    std::size_t temporary_bytes = 0;


    if (
        cub::DeviceScan::ExclusiveSum(
            temporary_storage,
            temporary_bytes,
            device_counts,
            device_offsets,
            count
        ) != cudaSuccess
    ) {
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    if (
        cudaMalloc(
            &temporary_storage,
            temporary_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    if (
        cub::DeviceScan::ExclusiveSum(
            temporary_storage,
            temporary_bytes,
            device_counts,
            device_offsets,
            count
        ) != cudaSuccess
    ) {
        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    write_final_prefix_offset_kernel
        <<<1, 1>>>(
            device_counts,
            count,
            device_offsets
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    offsets.resize(
        counts.size() + 1
    );


    if (
        cudaMemcpy(
            offsets.data(),
            device_offsets,
            offset_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        offsets.clear();

        cudaFree(temporary_storage);
        cudaFree(device_offsets);
        cudaFree(device_counts);

        return false;
    }


    cudaFree(temporary_storage);
    cudaFree(device_offsets);
    cudaFree(device_counts);


    return true;
}

bool cuda_expand_triplet_neighbor_records(
    const GpuFlatTripletIndexDevice& device_index,
    const std::vector<GpuTripletKey>& queries,
    const std::vector<std::uint32_t>& candidate_offsets,
    std::vector<GpuTripletCandidate>& candidates
)
{
    candidates.clear();


    if (
        candidate_offsets.size() !=
        queries.size() + 1
    ) {
        return false;
    }


    if (queries.empty()) {

        return (
            candidate_offsets.size() == 1 &&
            candidate_offsets[0] == 0
        );
    }


    if (
        queries.size() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        return false;
    }


    for (
        std::size_t i = 1;
        i < candidate_offsets.size();
        ++i
    ) {
        if (
            candidate_offsets[i] <
            candidate_offsets[i - 1]
        ) {
            return false;
        }
    }


    if (
        device_index.keys == nullptr ||
        device_index.offsets == nullptr ||
        (
            device_index.record_count > 0 &&
            device_index.records == nullptr
        )
    ) {
        return false;
    }


    const std::uint32_t total_candidates =
        candidate_offsets.back();


    if (total_candidates == 0)
        return true;


    GpuTripletKey* device_queries =
        nullptr;

    std::uint32_t* device_candidate_offsets =
        nullptr;

    GpuTripletCandidate* device_candidates =
        nullptr;


    const std::size_t query_bytes =
        queries.size() *
        sizeof(GpuTripletKey);

    const std::size_t offset_bytes =
        candidate_offsets.size() *
        sizeof(std::uint32_t);

    const std::size_t candidate_bytes =
        static_cast<std::size_t>(
            total_candidates
        ) *
        sizeof(GpuTripletCandidate);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_queries
            ),
            query_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_candidate_offsets
            ),
            offset_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_queries);
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_candidates
            ),
            candidate_bytes
        ) != cudaSuccess
    ) {
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    if (
        cudaMemcpy(
            device_queries,
            queries.data(),
            query_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    if (
        cudaMemcpy(
            device_candidate_offsets,
            candidate_offsets.data(),
            offset_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    const std::uint32_t query_count =
        static_cast<std::uint32_t>(
            queries.size()
        );


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            query_count +
            block_size - 1
        ) / block_size;


    triplet_neighbor_expand_kernel
        <<<block_count, block_size>>>(
            device_index.keys,
            device_index.offsets,
            device_index.records,
            device_index.key_count,
            device_queries,
            query_count,
            device_candidate_offsets,
            device_candidates
        );


    if (cudaGetLastError() != cudaSuccess) {

        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    if (cudaDeviceSynchronize() != cudaSuccess) {

        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    candidates.resize(
        total_candidates
    );


    if (
        cudaMemcpy(
            candidates.data(),
            device_candidates,
            candidate_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        candidates.clear();

        cudaFree(device_candidates);
        cudaFree(device_candidate_offsets);
        cudaFree(device_queries);

        return false;
    }


    cudaFree(device_candidates);
    cudaFree(device_candidate_offsets);
    cudaFree(device_queries);


    return true;
}