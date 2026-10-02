// src/interaction_areas.cpp
#include "interaction_areas.hpp"

#include <interaction/find_interaction_area.hpp>

std::vector<interaction::interaction_area>
get_protein_areas(RDKit::RWMol& protein)
{
    return interaction::find_protein_areas(protein);
}

std::vector<interaction::interaction_area>
get_ligand_areas(RDKit::RWMol& ligand)
{
    return interaction::find_ligand_areas(ligand);
}