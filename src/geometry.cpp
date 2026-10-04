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

interaction::point3D transform_point(
    const interaction::point3D& point,
    const RigidTransform& transform)
{
    const double x =
        static_cast<double>(point.x);

    const double y =
        static_cast<double>(point.y);

    const double z =
        static_cast<double>(point.z);

    return interaction::point3D{
        static_cast<interaction::coordinate_type>(
            transform.rotation[0] * x +
            transform.rotation[1] * y +
            transform.rotation[2] * z +
            transform.translation[0]
        ),

        static_cast<interaction::coordinate_type>(
            transform.rotation[3] * x +
            transform.rotation[4] * y +
            transform.rotation[5] * z +
            transform.translation[1]
        ),

        static_cast<interaction::coordinate_type>(
            transform.rotation[6] * x +
            transform.rotation[7] * y +
            transform.rotation[8] * z +
            transform.translation[2]
        )
    };
}


std::vector<MatchingArea> transform_matching_areas(
    const std::vector<MatchingArea>& areas,
    const RigidTransform& transform)
{
    std::vector<MatchingArea> transformed =
        areas;

    for (auto& area : transformed) {

        area.centroid =
            transform_point(
                area.centroid,
                transform
            );
    }

    return transformed;
}