#ifndef MOLECULE_IO_HPP
#define MOLECULE_IO_HPP

#include <memory>
#include <string>

#include <GraphMol/RWMol.h>

std::unique_ptr<RDKit::RWMol> load_protein(const std::string& path);
std::unique_ptr<RDKit::RWMol> load_pocket(const std::string& path);
std::unique_ptr<RDKit::RWMol> load_ligand(const std::string& path);

#endif // MOLECULE_IO_HPP