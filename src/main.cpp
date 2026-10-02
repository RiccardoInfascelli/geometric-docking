#include "molecule_io.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    try {
        const std::filesystem::path complex_dir =
            "/workspace/datasets/refined-set/1a1e";

        const std::string complex_id = "1a1e";

        const auto protein =
            load_protein(
                (complex_dir /
                 (complex_id + "_protein.pdb")).string()
            );

        const auto ligand =
            load_ligand(
                (complex_dir /
                 (complex_id + "_ligand.mol2")).string()
            );

        std::cout
            << "Protein atoms: "
            << protein->getNumAtoms()
            << "\n";

        std::cout
            << "Ligand atoms: "
            << ligand->getNumAtoms()
            << "\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "Error: "
            << e.what()
            << "\n";

        return 1;
    }
}