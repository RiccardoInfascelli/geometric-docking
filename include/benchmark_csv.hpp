#ifndef BENCHMARK_CSV_HPP
#define BENCHMARK_CSV_HPP

#include "pipeline.hpp"

#include <filesystem>

void append_benchmark_csv(
    const RunResult& result,
    const std::filesystem::path& csv_path
);

#endif