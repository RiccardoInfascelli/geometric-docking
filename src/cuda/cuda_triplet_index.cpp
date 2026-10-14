#include "cuda_triplet_index.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>


namespace
{

GpuTripletKey make_gpu_triplet_key(
    const TripletKey& key
)
{
    GpuTripletKey gpu_key{};


    for (
        std::size_t i = 0;
        i < 3;
        ++i
    ) {
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


    return gpu_key;
}

} // namespace


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

        result.keys.push_back(
            make_gpu_triplet_key(
                key
            )
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


        for (
            std::size_t i = 0;
            i < 3;
            ++i
        ) {
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


    for (
        std::size_t i = 0;
        i < cpu_index.keys.size();
        ++i
    ) {
        for (
            std::size_t j = 0;
            j < 3;
            ++j
        ) {
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


    for (
        std::size_t i = 0;
        i < cpu_index.offsets.size();
        ++i
    ) {
        if (
            gpu_index.offsets[i] !=
            cpu_index.offsets[i]
        ) {
            return false;
        }
    }


    for (
        std::size_t i = 0;
        i < cpu_index.records.size();
        ++i
    ) {
        for (
            std::size_t j = 0;
            j < 3;
            ++j
        ) {
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


void make_gpu_ligand_triplet_queries(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders
)
{
    queries.clear();
    orders.clear();


    queries.reserve(
        triplets.size()
    );

    orders.reserve(
        triplets.size()
    );


    for (const auto& triplet :
         triplets) {

        const auto canonical =
            canonicalize_triplet(
                triplet,
                bin_size
            );


        queries.push_back(
            make_gpu_triplet_key(
                make_triplet_key(
                    canonical
                )
            )
        );


        if (
            canonical.equivalent_order_count >
            6
        ) {
            throw std::runtime_error(
                "Too many equivalent ligand triplet orders"
            );
        }


        GpuLigandTripletOrders gpu_orders{};

        gpu_orders.equivalent_order_count =
            static_cast<std::uint32_t>(
                canonical.equivalent_order_count
            );


        for (
            std::size_t order_index = 0;
            order_index <
                canonical.equivalent_order_count;
            ++order_index
        ) {
            for (
                std::size_t vertex = 0;
                vertex < 3;
                ++vertex
            ) {
                const auto area_index =
                    canonical
                        .equivalent_orders[
                            order_index
                        ][vertex];


                if (
                    area_index >
                    std::numeric_limits<
                        std::uint32_t
                    >::max()
                ) {
                    throw std::overflow_error(
                        "Ligand matching-area index does not fit in uint32_t"
                    );
                }


                gpu_orders
                    .equivalent_orders[
                        order_index
                    ][vertex] =
                        static_cast<std::uint32_t>(
                            area_index
                        );
            }
        }


        orders.push_back(
            gpu_orders
        );
    }
}
