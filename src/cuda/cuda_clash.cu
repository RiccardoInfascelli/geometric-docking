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
    /*
     * Match the serial numerical semantics.
     *
     * interaction::coordinate_type is float:
     *
     * RDKit coordinates
     *   -> float input point
     *   -> double transform arithmetic
     *   -> float transformed point
     *   -> grid lookup
     *
     * GpuPoint3 remains double elsewhere so Kabsch itself
     * does not lose precision.
     */
    const double x =
        static_cast<double>(
            static_cast<float>(point.x)
        );

    const double y =
        static_cast<double>(
            static_cast<float>(point.y)
        );

    const double z =
        static_cast<double>(
            static_cast<float>(point.z)
        );


    const double transformed_x =
        transform.rotation[0] * x +
        transform.rotation[1] * y +
        transform.rotation[2] * z +
        transform.translation[0];

    const double transformed_y =
        transform.rotation[3] * x +
        transform.rotation[4] * y +
        transform.rotation[5] * z +
        transform.translation[1];

    const double transformed_z =
        transform.rotation[6] * x +
        transform.rotation[7] * y +
        transform.rotation[8] * z +
        transform.translation[2];


    GpuPoint3 result;

    result.x =
        static_cast<double>(
            static_cast<float>(
                transformed_x
            )
        );

    result.y =
        static_cast<double>(
            static_cast<float>(
                transformed_y
            )
        );

    result.z =
        static_cast<double>(
            static_cast<float>(
                transformed_z
            )
        );


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


bool cuda_score_grid_clash_device(
    const GpuPointArrayDevice& ligand_atoms,
    const GpuProteinGridDevice& protein_grid,
    const GpuRigidTransformArrayDevice& transforms,
    GpuClashScoreArrayDevice& scores
)
{
    cuda_free_clash_scores(
        scores
    );


    if (transforms.count == 0)
        return true;


    if (transforms.transforms == nullptr)
        return false;


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


    GpuClashScore* device_scores =
        nullptr;


    const std::size_t score_bytes =
        static_cast<std::size_t>(
            transforms.count
        ) *
        sizeof(GpuClashScore);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_scores
            ),
            score_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    constexpr std::uint32_t block_size =
        256;


    const std::uint32_t block_count =
        (
            transforms.count +
            block_size - 1
        ) /
        block_size;


    grid_clash_kernel
        <<<block_count, block_size>>>(
            ligand_atoms.points,
            ligand_atoms.count,
            protein_grid,
            transforms.transforms,
            transforms.count,
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

        return false;
    }


    /*
     * Keep clash scores resident on the GPU.
     * Ownership is transferred to the caller.
     */
    scores.scores =
        device_scores;

    scores.count =
        transforms.count;


    return true;
}


bool cuda_download_clash_scores(
    const GpuClashScoreArrayDevice& device_scores,
    std::vector<GpuClashScore>& host_scores
)
{
    host_scores.clear();


    if (device_scores.count == 0)
        return true;


    if (device_scores.scores == nullptr)
        return false;


    host_scores.resize(
        device_scores.count
    );


    const std::size_t bytes =
        static_cast<std::size_t>(
            device_scores.count
        ) *
        sizeof(GpuClashScore);


    if (
        cudaMemcpy(
            host_scores.data(),
            device_scores.scores,
            bytes,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess
    ) {
        host_scores.clear();

        return false;
    }


    return true;
}


void cuda_free_clash_scores(
    GpuClashScoreArrayDevice& device_scores
)
{
    if (device_scores.scores != nullptr) {

        cudaFree(
            device_scores.scores
        );
    }


    device_scores.scores =
        nullptr;

    device_scores.count =
        0;
}


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
        transforms.size() >
        static_cast<std::size_t>(
            std::numeric_limits<
                std::uint32_t
            >::max()
        )
    ) {
        return false;
    }


    GpuRigidTransformArrayDevice
        device_transforms;


    const std::size_t transform_bytes =
        transforms.size() *
        sizeof(GpuRigidTransform);


    if (
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_transforms.transforms
            ),
            transform_bytes
        ) != cudaSuccess
    ) {
        return false;
    }


    device_transforms.count =
        static_cast<std::uint32_t>(
            transforms.size()
        );


    if (
        cudaMemcpy(
            device_transforms.transforms,
            transforms.data(),
            transform_bytes,
            cudaMemcpyHostToDevice
        ) != cudaSuccess
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    GpuClashScoreArrayDevice
        device_scores;


    if (
        !cuda_score_grid_clash_device(
            ligand_atoms,
            protein_grid,
            device_transforms,
            device_scores
        )
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    const bool download_ok =
        cuda_download_clash_scores(
            device_scores,
            scores
        );


    cuda_free_clash_scores(
        device_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    return download_ok;
}