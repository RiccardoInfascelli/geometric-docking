#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP

#include "cuda_triplet_types.hpp"


bool cuda_smoke_test();


bool cuda_upload_triplet_index(
    const GpuFlatTripletIndexHost& host_index,
    GpuFlatTripletIndexDevice& device_index
);


void cuda_free_triplet_index(
    GpuFlatTripletIndexDevice& device_index
);


#endif