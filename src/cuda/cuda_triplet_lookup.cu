#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>


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