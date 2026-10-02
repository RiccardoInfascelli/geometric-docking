#include "molecule_io.hpp"

#include <interaction/sanitize_mol.hpp>

#include <GraphMol/FileParsers/FileParsers.h>

#include <fstream>
#include <sstream>
#include <stdexcept>


std::unique_ptr<RDKit::RWMol> load_protein(const std::string& path)
{
    std::ifstream file(path);

    if (!file)
        throw std::runtime_error("Cannot open protein file: " + path);

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::unique_ptr<RDKit::RWMol> molecule(
        RDKit::PDBBlockToMol(buffer.str(), false, false)
    );

    if (!molecule)
        throw std::runtime_error("Cannot parse protein file: " + path);

    interaction::sanitize_molecule(molecule);

    return molecule;
}


std::unique_ptr<RDKit::RWMol> load_ligand(const std::string& path)
{
    std::ifstream file(path);

    if (!file)
        throw std::runtime_error("Cannot open ligand file: " + path);

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string mol2 = buffer.str();

    std::unique_ptr<RDKit::RWMol> molecule(
        RDKit::Mol2BlockToMol(mol2, false, false)
    );

    if (!molecule)
        throw std::runtime_error("Cannot parse ligand file: " + path);

    interaction::sanitize_molecule(molecule);

    return molecule;
}