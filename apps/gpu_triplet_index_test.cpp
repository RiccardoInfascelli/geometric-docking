#include "cuda_triplet_index.hpp"

#include <iostream>

int main()
{
    FlatTripletIndex cpu_index;

    TripletKey key{};

    key.keys = {
        InteractionKey::Hydrophobic,
        InteractionKey::HydrogenDonor,
        InteractionKey::Aromatic
    };

    key.distance_bins = {
        4,
        7,
        10
    };

    cpu_index.keys.push_back(key);

    cpu_index.offsets = {
        0,
        2
    };

    cpu_index.records.push_back({
        {1, 2, 3}
    });

    cpu_index.records.push_back({
        {4, 5, 6}
    });

    const auto gpu_index =
        make_gpu_flat_triplet_index(
            cpu_index
        );

    if (
        !validate_gpu_flat_triplet_index(
            cpu_index,
            gpu_index
        )
    ) {
        std::cerr
            << "GPU triplet index conversion failed\n";

        return 1;
    }

    std::cout
        << "GPU triplet index conversion passed\n"
        << "keys: " << gpu_index.keys.size() << '\n'
        << "offsets: " << gpu_index.offsets.size() << '\n'
        << "records: " << gpu_index.records.size() << '\n';

    return 0;
}
