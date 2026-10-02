#ifndef MOLECULE_IO_HPP
#define MOLECULE_IO_HPP

#include <GraphMol/RWMol.h>

#include <memory>
#include <string>

std::unique_ptr<RDKit::RWMol>
load_protein(const std::string& path);

std::unique_ptr<RDKit::RWMol>
load_ligand(const std::string& path);

#endif