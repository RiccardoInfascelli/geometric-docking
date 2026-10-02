#include "molecule_io.hpp"
#include "interaction_areas.hpp"
#include "geometry.hpp"
#include "matching_areas.hpp"

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

        const auto protein_areas =
            get_protein_areas(*protein);

        const auto ligand_areas =
            get_ligand_areas(*ligand);

        std::cout
            << "Protein interaction areas: "
            << protein_areas.size()
            << "\n";

        std::cout
            << "Ligand interaction areas: "
            << ligand_areas.size()
            << "\n";

        const double ligand_diameter =
            compute_ligand_diameter(ligand_areas);

        std::cout
            << "Ligand diameter: "
            << ligand_diameter
            << " A\n";

        const auto ligand_matching_areas =
            build_ligand_matching_areas(ligand_areas);

        const auto protein_matching_areas =
            build_protein_matching_areas(protein_areas);

        std::cout
            << "Ligand matching areas: "
            << ligand_matching_areas.size()
            << "\n";

        std::cout
            << "Protein matching areas: "
            << protein_matching_areas.size()
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