#ifndef RUN_CONFIG_HPP
#define RUN_CONFIG_HPP

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>

enum class OutputMode
{
    Quiet,
    Summary,
    Debug
};

struct RunConfig
{
    // Input
    std::filesystem::path complex_directory;
    std::string complex_id;

    std::filesystem::path protein_path;
    std::filesystem::path ligand_path;
    std::filesystem::path pocket_path;

    // Output
    std::filesystem::path output_directory =
        "results";

    OutputMode output_mode =
        OutputMode::Summary;

    // Grid
    double grid_voxel_size =
        0.5;

    double grid_margin =
        6.5;

    // Triplets
    double triplet_bin_size =
        0.5;

    // Pocket search region
    double pocket_centroid_distance =
        2.0;

    // Clash
    double operational_clash_threshold =
        0.05;

    std::array<double, 4> clash_thresholds{
        0.00,
        0.025,
        0.05,
        0.10
    };

    // Ranking
    std::size_t fast_prefilter_top_n =
        500;
};

RunConfig parse_run_config(
    int argc,
    char* argv[]
);

#endif