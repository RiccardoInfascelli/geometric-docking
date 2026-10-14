#ifndef CUDA_TRIPLET_INDEX_HPP
#define CUDA_TRIPLET_INDEX_HPP

#include "cuda_triplet_types.hpp"
#include "triplet_generation.hpp"

#include <vector>


GpuFlatTripletIndexHost
make_gpu_flat_triplet_index(
    const FlatTripletIndex& index
);


bool validate_gpu_flat_triplet_index(
    const FlatTripletIndex& cpu_index,
    const GpuFlatTripletIndexHost& gpu_index
);


void make_gpu_ligand_triplet_queries(
    const std::vector<Triplet>& triplets,
    double bin_size,
    std::vector<GpuTripletKey>& queries,
    std::vector<GpuLigandTripletOrders>& orders
);


#endif
