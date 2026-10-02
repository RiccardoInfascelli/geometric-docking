#ifndef INTERACTION_AREAS_HPP
#define INTERACTION_AREAS_HPP

#include <GraphMol/RWMol.h>
#include <interaction/interaction_area.hpp>

#include <vector>

std::vector<interaction::interaction_area>
get_protein_areas(RDKit::RWMol& protein);

std::vector<interaction::interaction_area>
get_ligand_areas(RDKit::RWMol& ligand);


#endif