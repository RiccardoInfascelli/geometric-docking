#include "cuda_backend.hpp"

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


    host_index.keys = {
        make_key(0, 1, 2, 1, 2, 3),
        make_key(0, 1, 2, 4, 5, 6),
        make_key(1, 2, 3, 7, 8, 9)
    };


    host_index.offsets = {
        0,
        1,
        2,
        3
    };


    host_index.records = {
        {{0, 1, 2}},
        {{3, 4, 5}},
        {{6, 7, 8}}
    };


    GpuFlatTripletIndexDevice device_index;


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


    const std::vector<GpuTripletKey> queries = {
        host_index.keys[0],
        host_index.keys[2],

        make_key(
            0, 1, 2,
            100, 100, 100
        )
    };


    std::vector<std::uint32_t> results;


    const bool lookup_ok =
        cuda_lookup_triplet_keys_exact(
            device_index,
            queries,
            results
        );


    cuda_free_triplet_index(
        device_index
    );


    if (!lookup_ok) {

        std::cerr
            << "Exact CUDA lookup failed\n";

        return 1;
    }


    if (
        results.size() != 3 ||
        results[0] != 0 ||
        results[1] != 2 ||
        results[2] !=
            GPU_TRIPLET_KEY_NOT_FOUND
    ) {
        std::cerr
            << "Unexpected lookup results\n";

        return 1;
    }


    std::cout
        << "CUDA exact triplet lookup passed\n";

    return 0;
}