#include "geometry.hpp"

#include <interaction/geometric_features.hpp>

#include <cmath>

double compute_ligand_diameter(
    const std::vector<interaction::interaction_area>& ligand_areas)
{
    double max_distance = 0.0;

    for (std::size_t i = 0; i < ligand_areas.size(); ++i) {

        for (std::size_t j = i + 1;
             j < ligand_areas.size();
             ++j) {

            const double distance =
                std::sqrt(
                    interaction::distance2(
                        ligand_areas[i].centroid,
                        ligand_areas[j].centroid
                    )
                );

            if (distance > max_distance)
                max_distance = distance;
        }
    }

    return max_distance;
}