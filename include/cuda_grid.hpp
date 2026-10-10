#ifndef CUDA_GRID_HPP
#define CUDA_GRID_HPP

#include "protein_grid.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>


struct GpuGridVoxel
{
    std::uint16_t interactions{};
    std::uint8_t occupied{};
    std::uint8_t padding{};
};


struct GpuProteinGridHost
{
    double xmin{};
    double ymin{};
    double zmin{};

    double voxel_size{};

    std::int32_t nx{};
    std::int32_t ny{};
    std::int32_t nz{};

    std::vector<GpuGridVoxel> voxels;
};


static_assert(
    std::is_trivially_copyable_v<
        GpuGridVoxel
    >
);

static_assert(
    sizeof(GpuGridVoxel) == 4
);


GpuProteinGridHost make_gpu_protein_grid(
    const ProteinGrid& grid
);


bool validate_gpu_protein_grid(
    const ProteinGrid& cpu_grid,
    const GpuProteinGridHost& gpu_grid
);


#endif