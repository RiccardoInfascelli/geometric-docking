#ifndef POSE_OUTPUT_HPP
#define POSE_OUTPUT_HPP

#include "pipeline.hpp"
#include "run_config.hpp"

#include <cstddef>

void save_pose_outputs(
    const RunResult& result,
    const RunConfig& config,
    std::size_t top_n = 20
);

#endif