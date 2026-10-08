#ifndef TRIPLET_GENERATION_HPP
#define TRIPLET_GENERATION_HPP

#include "matching_areas.hpp"
#include "matching_grid.hpp"
#include "pose_targets.hpp"

#include <array>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <unordered_map>


struct Triplet
{
    std::array<std::size_t, 3> matching_area_indices;
    std::array<InteractionKey, 3> keys;
    std::array<double, 3> distances;
};

std::vector<Triplet> generate_triplets(
    const std::vector<MatchingArea>& areas,
    double max_distance
);

std::size_t count_triplets_spatial(
    const std::vector<MatchingArea>& areas,
    const MatchingGrid& grid,
    double max_distance
);

struct CanonicalTriplet
{
    std::array<std::size_t, 3> matching_area_indices{};

    std::array<InteractionKey, 3> keys{};

    std::array<double, 3> distances{};

    std::array<std::size_t, 3> distance_bins{};

    /*
     * Tutti gli ordini che producono
     * la stessa firma canonica.
     *
     * Caso normale: 1
     * Simmetria parziale: 2
     * Simmetria completa: fino a 6
     */
    std::array<
        std::array<std::size_t, 3>,
        6
    > equivalent_orders{};

    std::uint8_t equivalent_order_count = 0;
};


CanonicalTriplet canonicalize_triplet(
    const Triplet& triplet,
    double bin_size
);

struct TripletKey
{
    std::array<InteractionKey, 3> keys{};
    std::array<std::size_t, 3> distance_bins{};

    bool operator==(
        const TripletKey& other
    ) const = default;
};


struct TripletKeyHash
{
    std::size_t operator()(
        const TripletKey& key
    ) const;
};


TripletKey make_triplet_key(
    const CanonicalTriplet& triplet
);

struct ProteinTripletRecord
{
    std::array<std::uint32_t, 3> indices{};
};


using ProteinTripletHashTable =
    std::unordered_map<
        TripletKey,
        std::vector<ProteinTripletRecord>,
        TripletKeyHash
    >;

ProteinTripletRecord make_protein_triplet_record(
    const CanonicalTriplet& triplet
);

ProteinTripletHashTable build_protein_triplet_hash_table(
    const std::vector<PoseTarget>& targets,
    double max_distance,
    double bin_size
);

struct FlatTripletIndex
{
    std::vector<TripletKey> keys;

    std::vector<std::size_t> offsets;

    std::vector<ProteinTripletRecord> records;
};

struct FlatTripletBucket
{
    std::size_t begin{};
    std::size_t end{};
};


std::vector<FlatTripletBucket>
find_neighbor_triplet_buckets_flat(
    const FlatTripletIndex& index,
    const CanonicalTriplet& ligand_triplet
);


FlatTripletIndex flatten_protein_triplet_index(
    const ProteinTripletHashTable& table
);

std::vector<const std::vector<ProteinTripletRecord>*>
find_neighbor_triplet_buckets(
    const ProteinTripletHashTable& table,
    const CanonicalTriplet& ligand_triplet
);

bool validate_flat_triplet_index(
    const ProteinTripletHashTable& table,
    const FlatTripletIndex& flat_index
);

#endif