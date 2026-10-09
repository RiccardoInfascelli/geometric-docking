#include "cuda_triplet_index.hpp"

#include <limits>
#include <stdexcept>


GpuFlatTripletIndexHost
make_gpu_flat_triplet_index(
    const FlatTripletIndex& index
)
{
    GpuFlatTripletIndexHost result;


    result.keys.reserve(
        index.keys.size()
    );

    result.offsets.reserve(
        index.offsets.size()
    );

    result.records.reserve(
        index.records.size()
    );


    for (const auto& key :
         index.keys) {

        GpuTripletKey gpu_key{};

        for (std::size_t i = 0;
             i < 3;
             ++i) {

            const auto interaction_value =
                static_cast<unsigned int>(
                    key.keys[i]
                );

            if (
                interaction_value >
                std::numeric_limits<
                    std::uint8_t
                >::max()
            ) {
                throw std::overflow_error(
                    "InteractionKey does not fit in uint8_t"
                );
            }


            if (
                key.distance_bins[i] >
                std::numeric_limits<
                    std::uint32_t
                >::max()
            ) {
                throw std::overflow_error(
                    "Triplet distance bin does not fit in uint32_t"
                );
            }


            gpu_key.keys[i] =
                static_cast<std::uint8_t>(
                    interaction_value
                );

            gpu_key.distance_bins[i] =
                static_cast<std::uint32_t>(
                    key.distance_bins[i]
                );
        }


        result.keys.push_back(
            gpu_key
        );
    }


    for (const auto offset :
         index.offsets) {

        if (
            offset >
            std::numeric_limits<
                std::uint32_t
            >::max()
        ) {
            throw std::overflow_error(
                "Triplet index offset does not fit in uint32_t"
            );
        }

        result.offsets.push_back(
            static_cast<std::uint32_t>(
                offset
            )
        );
    }


    for (const auto& record :
         index.records) {

        GpuProteinTripletRecord gpu_record{};

        for (std::size_t i = 0;
             i < 3;
             ++i) {

            gpu_record.indices[i] =
                record.indices[i];
        }

        result.records.push_back(
            gpu_record
        );
    }


    return result;
}


bool validate_gpu_flat_triplet_index(
    const FlatTripletIndex& cpu_index,
    const GpuFlatTripletIndexHost& gpu_index
)
{
    if (
        cpu_index.keys.size() !=
            gpu_index.keys.size()
        ||
        cpu_index.offsets.size() !=
            gpu_index.offsets.size()
        ||
        cpu_index.records.size() !=
            gpu_index.records.size()
    ) {
        return false;
    }


    for (std::size_t i = 0;
         i < cpu_index.keys.size();
         ++i) {

        for (std::size_t j = 0;
             j < 3;
             ++j) {

            if (
                gpu_index.keys[i].keys[j] !=
                static_cast<std::uint8_t>(
                    cpu_index.keys[i].keys[j]
                )
            ) {
                return false;
            }


            if (
                gpu_index.keys[i]
                    .distance_bins[j] !=
                cpu_index.keys[i]
                    .distance_bins[j]
            ) {
                return false;
            }
        }
    }


    for (std::size_t i = 0;
         i < cpu_index.offsets.size();
         ++i) {

        if (
            gpu_index.offsets[i] !=
            cpu_index.offsets[i]
        ) {
            return false;
        }
    }


    for (std::size_t i = 0;
         i < cpu_index.records.size();
         ++i) {

        for (std::size_t j = 0;
             j < 3;
             ++j) {

            if (
                gpu_index.records[i]
                    .indices[j] !=
                cpu_index.records[i]
                    .indices[j]
            ) {
                return false;
            }
        }
    }


    return true;
}