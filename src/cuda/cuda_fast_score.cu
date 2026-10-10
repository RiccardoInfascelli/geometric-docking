#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>


namespace
{

__device__ bool grid_in_bounds(
    const GpuProteinGridDevice& grid,
    int x,
    int y,
    int z
)
{
    return
        x >= 0 &&
        x < grid.nx &&
        y >= 0 &&
        y < grid.ny &&
        z >= 0 &&
        z < grid.nz;
}


__device__ std::size_t grid_index(
    const GpuProteinGridDevice& grid,
    int x,
    int y,
    int z
)
{
    return static_cast<std::size_t>(
        x +
        grid.nx *
        (
            y +
            grid.ny * z
        )
    );
}


__device__ int to_grid(
    double coordinate,
    double minimum,
    double voxel_size
)
{
    return static_cast<int>(
        floor(
            (
                coordinate -
                minimum
            ) /
            voxel_size
        )
    );
}


__device__ GpuPoint3 transform_point(
    const GpuPoint3& point,
    const GpuRigidTransform& transform
)
{
    GpuPoint3 result;


    result.x =
        transform.rotation[0] * point.x +
        transform.rotation[1] * point.y +
        transform.rotation[2] * point.z +
        transform.translation[0];


    result.y =
        transform.rotation[3] * point.x +
        transform.rotation[4] * point.y +
        transform.rotation[5] * point.z +
        transform.translation[1];


    result.z =
        transform.rotation[6] * point.x +
        transform.rotation[7] * point.y +
        transform.rotation[8] * point.z +
        transform.translation[2];


    return result;
}


__global__ void fast_score_kernel(
    const GpuFastMatchingArea* ligand_areas,
    std::uint32_t ligand_area_count,
    GpuProteinGridDevice protein_grid,
    const GpuRigidTransform* transforms,
    std::uint32_t transform_count,
    GpuFastScore* scores
)
{
    const std::uint32_t pose_index =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;


    if (
        pose_index >=
        transform_count
    ) {
        return;
    }


    const auto transform =
        transforms[
            pose_index
        ];


    GpuFastScore score;


    score.total_areas =
        ligand_area_count;


    if (transform.valid == 0) {

        score.valid = 0;

        scores[
            pose_index
        ] = score;

        return;
    }


    score.valid = 1;


    for (
        std::uint32_t area_index = 0;
        area_index < ligand_area_count;
        ++area_index
    ) {
        const auto area =
            ligand_areas[
                area_index
            ];


        const auto transformed =
            transform_point(
                area.point,
                transform
            );


        const int gx =
            to_grid(
                transformed.x,
                protein_grid.xmin,
                protein_grid.voxel_size
            );


        const int gy =
            to_grid(
                transformed.y,
                protein_grid.ymin,
                protein_grid.voxel_size
            );


        const int gz =
            to_grid(
                transformed.z,
                protein_grid.zmin,
                protein_grid.voxel_size
            );


        if (
            !grid_in_bounds(
                protein_grid,
                gx,
                gy,
                gz
            )
        ) {
            continue;
        }


        ++score.areas_in_grid;


        const std::size_t voxel_index =
            grid_index(
                protein_grid,
                gx,
                gy,
                gz
            );


        const auto voxel =
            protein_grid
                .voxels[
                    voxel_index
                ];


        if (voxel.occupied != 0) {

            ++score
                .areas_on_occupied_voxel;

            continue;
        }


        if (
            (
                voxel.interactions &
                area.interaction_bit
            ) == 0
        ) {
            continue;
        }


        ++score.compatible_areas;


        score.weighted_score +=
            area.weight;
    }


    scores[
        pose_index
    ] = score;
}

} // namespace


void cuda_free_fast_matching_areas(
    GpuFastMatchingAreaArrayDevice& device_areas
)
{
    if (
        device_areas.areas !=
        nullptr
    ) {
        cudaFree(
            device_areas.areas
        );
    }


    device_areas =
        GpuFastMatchingAreaArrayDevice{};
}


bool cuda_upload_fast_matching_areas(
    const std::vector<GpuFastMatchingArea>& host_areas,
    GpuFastMatchingAreaArrayDevice& device_areas
)
{
    if (
        host_areas.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    if (host_areas.empty()) {

        cuda_free_fast_matching_areas(
            device_areas
        );

        return true;
    }


    GpuFastMatchingAreaArrayDevice
        replacement;


    const std::size_t bytes =
        host_areas.size() *
        sizeof(GpuFastMatchingArea);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &replacement.areas
            ),
            bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMemcpy(
            replacement.areas,
            host_areas.data(),
            bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_fast_matching_areas(
            replacement
        );

        return false;
    }


    replacement.count =
        static_cast<std::uint32_t>(
            host_areas.size()
        );


    cuda_free_fast_matching_areas(
        device_areas
    );


    device_areas =
        replacement;


    return true;
}


bool cuda_score_fast_interactions(
    const GpuFastMatchingAreaArrayDevice& ligand_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuFastScore>& scores
)
{
    scores.clear();


    if (transforms.empty())
        return true;


    if (
        ligand_areas.count != 0 &&
        ligand_areas.areas == nullptr
    ) {
        return false;
    }


    if (
        protein_grid.voxels == nullptr ||
        protein_grid.voxel_count == 0
    ) {
        return false;
    }


    if (
        transforms.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    GpuRigidTransform*
        device_transforms =
            nullptr;


    GpuFastScore*
        device_scores =
            nullptr;


    const std::size_t transform_bytes =
        transforms.size() *
        sizeof(GpuRigidTransform);


    const std::size_t score_bytes =
        transforms.size() *
        sizeof(GpuFastScore);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_transforms
            ),
            transform_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_scores
            ),
            score_bytes
        ) != cudaSuccess
    ) {
        cudaFree(
            device_transforms
        );

        return false;
    }


    if (
        cudaMemcpy(
            device_transforms,
            transforms.data(),
            transform_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cudaFree(
            device_scores
        );

        cudaFree(
            device_transforms
        );

        return false;
    }


    const std::uint32_t transform_count =
        static_cast<std::uint32_t>(
            transforms.size()
        );


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            transform_count +
            block_size - 1
        ) /
        block_size;


    fast_score_kernel
        <<<block_count, block_size>>>(
            ligand_areas.areas,
            ligand_areas.count,
            protein_grid,
            device_transforms,
            transform_count,
            device_scores
        );


    if (
        cudaGetLastError() !=
            cudaSuccess ||
        cudaDeviceSynchronize() !=
            cudaSuccess
    ) {
        cudaFree(
            device_scores
        );

        cudaFree(
            device_transforms
        );

        return false;
    }


    scores.resize(
        transforms.size()
    );


    if (
        cudaMemcpy(
            scores.data(),
            device_scores,
            score_bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        scores.clear();

        cudaFree(
            device_scores
        );

        cudaFree(
            device_transforms
        );

        return false;
    }


    cudaFree(
        device_scores
    );

    cudaFree(
        device_transforms
    );


    return true;
}