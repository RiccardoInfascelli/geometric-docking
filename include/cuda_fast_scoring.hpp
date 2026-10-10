#ifndef CUDA_FAST_SCORING_HPP
#define CUDA_FAST_SCORING_HPP

#include "cuda_triplet_types.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>


struct GpuFastMatchingArea
{
    GpuPoint3 point{};

    std::uint16_t interaction_bit{};
    std::uint16_t padding[3]{};

    double weight{};
};


static_assert(
    std::is_trivially_copyable_v<
        GpuFastMatchingArea
    >
);

static_assert(
    sizeof(GpuFastMatchingArea) == 40
);


std::vector<GpuFastMatchingArea>
make_gpu_fast_matching_areas(
    const std::vector<MatchingArea>& areas,
    const FastScoreWeights& weights
);


#endif