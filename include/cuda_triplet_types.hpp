#ifndef CUDA_TRIPLET_TYPES_HPP
#define CUDA_TRIPLET_TYPES_HPP

#include <cstdint>
#include <type_traits>
#include <vector>


struct GpuTripletKey
{
    std::uint8_t keys[3]{};
    std::uint8_t padding{};

    std::uint32_t distance_bins[3]{};
};


struct GpuProteinTripletRecord
{
    std::uint32_t indices[3]{};
};


struct GpuFlatTripletIndexHost
{
    std::vector<GpuTripletKey> keys;
    std::vector<std::uint32_t> offsets;
    std::vector<GpuProteinTripletRecord> records;
};


struct GpuFlatTripletIndexDevice
{
    GpuTripletKey* keys{nullptr};

    std::uint32_t* offsets{nullptr};

    GpuProteinTripletRecord* records{nullptr};

    std::uint32_t key_count{};
    std::uint32_t offset_count{};
    std::uint32_t record_count{};
};


static_assert(
    std::is_trivially_copyable_v<GpuTripletKey>
);

static_assert(
    std::is_trivially_copyable_v<
        GpuProteinTripletRecord
    >
);

static_assert(
    sizeof(GpuTripletKey) == 16
);

static_assert(
    sizeof(GpuProteinTripletRecord) == 12
);

struct GpuTripletCandidate
{
    std::uint32_t ligand_triplet_index{};
    std::uint32_t protein_indices[3]{};
};

static_assert(
    std::is_trivially_copyable_v<
        GpuTripletCandidate
    >
);

static_assert(
    sizeof(GpuTripletCandidate) == 16
);

#endif