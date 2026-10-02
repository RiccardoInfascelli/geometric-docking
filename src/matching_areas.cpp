#include "matching_areas.hpp"

namespace
{

void add_area(
    std::vector<MatchingArea>& result,
    std::size_t source_index,
    const interaction::point3D& centroid,
    InteractionKey key)
{
    result.push_back({
        source_index,
        centroid,
        key
    });
}

} // namespace


std::vector<MatchingArea> build_ligand_matching_areas(
    const std::vector<interaction::interaction_area>& areas)
{
    using P = interaction::interaction_smarts_pattern;
    using C = interaction::interaction_class_pattern;

    std::vector<MatchingArea> result;

    for (std::size_t i = 0; i < areas.size(); ++i) {

        const auto& area = areas[i];

        if (area.pattern_class == C::Aromatic) {
            add_area(result, i, area.centroid, InteractionKey::Aromatic);
            continue;
        }

        switch (area.pattern) {

            case P::Hydrophobic:
                add_area(result, i, area.centroid,
                         InteractionKey::Hydrophobic);
                break;

            case P::Hydrogen_donor:
                add_area(result, i, area.centroid,
                         InteractionKey::HydrogenDonor);
                break;

            case P::Hydrogen_acceptor:
                add_area(result, i, area.centroid,
                         InteractionKey::HydrogenAcceptor);
                break;

            case P::Halogen_donor:
                add_area(result, i, area.centroid,
                         InteractionKey::HalogenDonor);
                break;

            case P::Halogen_acceptor:
                add_area(result, i, area.centroid,
                         InteractionKey::HalogenAcceptor);
                break;

            case P::Cation:
                add_area(result, i, area.centroid,
                         InteractionKey::Cation);
                break;

            case P::Anion:
                add_area(result, i, area.centroid,
                         InteractionKey::Anion);
                break;

            case P::Metal:
                add_area(result, i, area.centroid,
                         InteractionKey::Metal);
                break;

            case P::Chelated:
                add_area(result, i, area.centroid,
                         InteractionKey::Chelated);
                break;

            default:
                break;
        }
    }

    return result;
}

std::vector<MatchingArea> build_protein_matching_areas(
    const std::vector<interaction::interaction_area>& areas)
{
    using P = interaction::interaction_smarts_pattern;
    using C = interaction::interaction_class_pattern;

    std::vector<MatchingArea> result;

    for (std::size_t i = 0; i < areas.size(); ++i) {

        const auto& area = areas[i];

        /*
         * Aromatic protein area:
         *
         * - ligand Aromatic -> pi stacking
         * - ligand Cation   -> cation-pi
         */
        if (area.pattern_class == C::Aromatic) {

            add_area(result, i, area.centroid,
                     InteractionKey::Aromatic);

            add_area(result, i, area.centroid,
                     InteractionKey::Cation);

            continue;
        }

        switch (area.pattern) {

            case P::Hydrophobic:
                add_area(result, i, area.centroid,
                         InteractionKey::Hydrophobic);
                break;

            /*
             * Protein donor matches ligand acceptor.
             */
            case P::Hydrogen_donor:
                add_area(result, i, area.centroid,
                         InteractionKey::HydrogenAcceptor);
                break;

            /*
             * Protein acceptor matches ligand donor.
             */
            case P::Hydrogen_acceptor:
                add_area(result, i, area.centroid,
                         InteractionKey::HydrogenDonor);
                break;

            case P::Halogen_donor:
                add_area(result, i, area.centroid,
                         InteractionKey::HalogenAcceptor);
                break;

            case P::Halogen_acceptor:
                add_area(result, i, area.centroid,
                         InteractionKey::HalogenDonor);
                break;

            /*
             * Protein anion matches ligand cation.
             */
            case P::Anion:
                add_area(result, i, area.centroid,
                         InteractionKey::Cation);
                break;

            /*
             * Protein cation can match:
             *
             * - ligand anion    -> ionic
             * - ligand aromatic -> cation-pi
             */
            case P::Cation:

                add_area(result, i, area.centroid,
                         InteractionKey::Anion);

                add_area(result, i, area.centroid,
                         InteractionKey::Aromatic);

                break;

            /*
             * Metal <-> Chelated
             */
            case P::Metal:
                add_area(result, i, area.centroid,
                         InteractionKey::Chelated);
                break;

            case P::Chelated:
                add_area(result, i, area.centroid,
                         InteractionKey::Metal);
                break;

            default:
                break;
        }
    }

    return result;
}