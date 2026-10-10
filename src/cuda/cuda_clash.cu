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
        transform.rotation[0] *
            point.x +
        transform.rotation[1] *
            point.y +
        transform.rotation[2] *
            point.z +
        transform.translation[0];


    result.y =
        transform.rotation[3] *
            point.x +
        transform.rotation[4] *
            point.y +
        transform.rotation[5] *
            point.z +
        transform.translation[1];


    result.z =
        transform.rotation[6] *
            point.x +
        transform.rotation[7] *
            point.y +
        transform.rotation[8] *
            point.z +
        transform.translation[2];


    return result;
}


__global__ void grid_clash_kernel(
    const GpuPoint3* ligand_atoms,
    std::uint32_t ligand_atom_count,
    GpuProteinGridDevice protein_grid,
    const GpuRigidTransform* transforms,
    std::uint32_t transform_count,
    GpuClashScore* scores
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


    GpuClashScore score;


    score.total_atoms =
        ligand_atom_count;


    if (transform.valid == 0) {

        score.valid = 0;

        scores[
            pose_index
        ] = score;

        return;
    }


    score.valid = 1;


    for (
        std::uint32_t atom_index = 0;
        atom_index < ligand_atom_count;
        ++atom_index
    ) {
        const auto transformed =
            transform_point(
                ligand_atoms[
                    atom_index
                ],
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


        /*
         * Same semantics as the serial
         * score_fast_grid_clash:
         *
         * atoms outside the grid do not
         * contribute to the clash count.
         */
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


        const std::size_t voxel_index =
            grid_index(
                protein_grid,
                gx,
                gy,
                gz
            );


        if (
            protein_grid
                .voxels[
                    voxel_index
                ]
                .occupied != 0
        ) {
            ++score.occupied_atoms;
        }
    }


    if (score.total_atoms != 0) {

        score.fraction =
            static_cast<double>(
                score.occupied_atoms
            ) /
            static_cast<double>(
                score.total_atoms
            );
    }


    scores[
        pose_index
    ] = score;
}

} // namespace


bool cuda_score_grid_clash(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuRigidTransform>& transforms,
    std::vector<GpuClashScore>& scores
)
{
    scores.clear();


    if (transforms.empty())
        return true;


    if (
        ligand_atoms.count != 0 &&
        ligand_atoms.points == nullptr
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
        static_cast<std::size_t>(
            std::numeric_limits<
                std::uint32_t
            >::max()
        )
    ) {
        return false;
    }


    GpuRigidTransform*
        device_transforms =
            nullptr;

    GpuClashScore*
        device_scores =
            nullptr;


    const std::size_t transform_bytes =
        transforms.size() *
        sizeof(GpuRigidTransform);


    const std::size_t score_bytes =
        transforms.size() *
        sizeof(GpuClashScore);


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


    grid_clash_kernel
        <<<block_count, block_size>>>(
            ligand_atoms.points,
            ligand_atoms.count,
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