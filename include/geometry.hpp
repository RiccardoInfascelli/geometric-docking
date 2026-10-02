#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <interaction/interaction_area.hpp>

#include <vector>

double compute_ligand_diameter(
    const std::vector<interaction::interaction_area>& ligand_areas
);

#endif