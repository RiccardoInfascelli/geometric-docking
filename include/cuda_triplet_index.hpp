#ifndef CUDA_TRIPLET_INDEX_HPP
#define CUDA_TRIPLET_INDEX_HPP

#include "cuda_triplet_types.hpp"
#include "triplet_generation.hpp"


GpuFlatTripletIndexHost
make_gpu_flat_triplet_index(
    const FlatTripletIndex& index
);


bool validate_gpu_flat_triplet_index(
    const FlatTripletIndex& cpu_index,
    const GpuFlatTripletIndexHost& gpu_index
);


#endif