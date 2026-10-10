#include "cuda_backend.hpp"

#include <cmath>
#include <iostream>
#include <vector>


namespace
{

GpuPoint3 transform_point(
    const GpuPoint3& point,
    const GpuRigidTransform& transform
)
{
    return {
        transform.rotation[0] * point.x +
        transform.rotation[1] * point.y +
        transform.rotation[2] * point.z +
        transform.translation[0],

        transform.rotation[3] * point.x +
        transform.rotation[4] * point.y +
        transform.rotation[5] * point.z +
        transform.translation[1],

        transform.rotation[6] * point.x +
        transform.rotation[7] * point.y +
        transform.rotation[8] * point.z +
        transform.translation[2]
    };
}


bool same_point(
    const GpuPoint3& a,
    const GpuPoint3& b
)
{
    constexpr double tolerance =
        1e-8;


    return
        std::abs(a.x - b.x) < tolerance &&
        std::abs(a.y - b.y) < tolerance &&
        std::abs(a.z - b.z) < tolerance;
}

} // namespace


int main()
{
    /*
     * Source triangle.
     */
    const std::vector<GpuPoint3>
        ligand_points = {

            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}
        };


    /*
     * First target:
     *
     * identity rotation +
     * translation (2,-3,4)
     *
     * Second target:
     *
     * 90-degree rotation around Z +
     * translation (5,6,7)
     */
    const std::vector<GpuPoint3>
        protein_points = {

            {2.0, -3.0, 4.0},
            {3.0, -3.0, 4.0},
            {2.0, -2.0, 4.0},

            {5.0, 6.0, 7.0},
            {5.0, 7.0, 7.0},
            {4.0, 6.0, 7.0}
        };


    GpuPointArrayDevice
        device_ligand_points;

    GpuPointArrayDevice
        device_protein_points;


    if (
        !cuda_upload_points(
            ligand_points,
            device_ligand_points
        )
    ) {
        std::cerr
            << "Ligand point upload failed\n";

        return 1;
    }


    if (
        !cuda_upload_points(
            protein_points,
            device_protein_points
        )
    ) {
        std::cerr
            << "Protein point upload failed\n";

        cuda_free_points(
            device_ligand_points
        );

        return 1;
    }


    const std::vector<GpuAlignmentCandidate>
        alignments = {

            {
                {0, 1, 2},
                {0, 1, 2}
            },

            {
                {0, 1, 2},
                {3, 4, 5}
            }
        };


    std::vector<GpuRigidTransform>
        transforms;


    if (
        !cuda_compute_kabsch_transforms(
            device_ligand_points,
            device_protein_points,
            alignments,
            transforms
        )
    ) {
        std::cerr
            << "CUDA Kabsch failed\n";

        cuda_free_points(
            device_protein_points
        );

        cuda_free_points(
            device_ligand_points
        );

        return 1;
    }


    if (
        transforms.size() != 2 ||
        transforms[0].valid == 0 ||
        transforms[1].valid == 0
    ) {
        std::cerr
            << "Invalid CUDA Kabsch transforms\n";

        cuda_free_points(
            device_protein_points
        );

        cuda_free_points(
            device_ligand_points
        );

        return 1;
    }


    /*
     * Check the transformations by applying
     * them to the original source points.
     */
    for (int i = 0; i < 3; ++i) {

        const auto transformed =
            transform_point(
                ligand_points[i],
                transforms[0]
            );


        if (
            !same_point(
                transformed,
                protein_points[i]
            )
        ) {
            std::cerr
                << "Translation Kabsch test failed\n";

            cuda_free_points(
                device_protein_points
            );

            cuda_free_points(
                device_ligand_points
            );

            return 1;
        }
    }


    std::cout
        << "CUDA translation Kabsch passed\n";


    for (int i = 0; i < 3; ++i) {

        const auto transformed =
            transform_point(
                ligand_points[i],
                transforms[1]
            );


        if (
            !same_point(
                transformed,
                protein_points[
                    3 + i
                ]
            )
        ) {
            std::cerr
                << "Rotation Kabsch test failed\n";

            cuda_free_points(
                device_protein_points
            );

            cuda_free_points(
                device_ligand_points
            );

            return 1;
        }
    }


    std::cout
        << "CUDA rotation Kabsch passed\n";


    cuda_free_points(
        device_protein_points
    );

    cuda_free_points(
        device_ligand_points
    );


    std::cout
        << "CUDA Kabsch tests passed\n";


    return 0;
}