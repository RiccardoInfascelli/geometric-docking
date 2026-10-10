#ifndef CUDA_GEOMETRY_HPP
#define CUDA_GEOMETRY_HPP

#include "cuda_triplet_types.hpp"
#include "matching_areas.hpp"
#include "pose_targets.hpp"

#include <vector>


std::vector<GpuPoint3>
make_gpu_ligand_matching_points(
    const std::vector<MatchingArea>& areas
);


std::vector<GpuPoint3>
make_gpu_pose_target_points(
    const std::vector<PoseTarget>& targets
);


#endif