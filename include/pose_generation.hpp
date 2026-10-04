#ifndef POSE_GENERATION_HPP
#define POSE_GENERATION_HPP

#include "geometry.hpp"

#include <array>

bool rigid_transform_from_points(
    const std::array<interaction::point3D, 3>& source,
    const std::array<interaction::point3D, 3>& target,
    RigidTransform& transform
);

#endif