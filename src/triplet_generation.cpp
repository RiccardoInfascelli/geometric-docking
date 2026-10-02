#include "triplet_generation.hpp"

#include <interaction/geometric_features.hpp>

#include <cmath>

namespace
{

InteractionKey to_ligand_key(
    interaction::interaction_smarts_pattern pattern,
    interaction::interaction_class_pattern pattern_class)
{
    using P = interaction::interaction_smarts_pattern;
    using PC = interaction::interaction_class_pattern;

    if (pattern_class == PC::Aromatic)
        return InteractionKey::Aromatic;

    switch (pattern) {
        case P::Hydrophobic:       return InteractionKey::Hydrophobic;
        case P::Hydrogen_donor:    return InteractionKey::HydrogenDonor;
        case P::Hydrogen_acceptor: return InteractionKey::HydrogenAcceptor;
        case P::Halogen_donor:     return InteractionKey::HalogenDonor;
        case P::Halogen_acceptor:  return InteractionKey::HalogenAcceptor;
        case P::Cation:            return InteractionKey::Cation;
        case P::Anion:             return InteractionKey::Anion;
        case P::Metal:             return InteractionKey::Metal;
        case P::Chelated:          return InteractionKey::Chelated;
        default:                   return InteractionKey::Unknown;
    }
}

InteractionKey to_protein_key(
    interaction::interaction_smarts_pattern pattern,
    interaction::interaction_class_pattern pattern_class)
{
    using P = interaction::interaction_smarts_pattern;
    using PC = interaction::interaction_class_pattern;

    if (pattern_class == PC::Aromatic)
        return InteractionKey::Aromatic;

    switch (pattern) {
        case P::Hydrophobic:       return InteractionKey::Hydrophobic;
        case P::Hydrogen_donor:    return InteractionKey::HydrogenAcceptor;
        case P::Hydrogen_acceptor: return InteractionKey::HydrogenDonor;
        case P::Halogen_donor:     return InteractionKey::HalogenAcceptor;
        case P::Halogen_acceptor:  return InteractionKey::HalogenDonor;
        case P::Cation:            return InteractionKey::Anion;
        case P::Anion:             return InteractionKey::Cation;
        case P::Metal:             return InteractionKey::Metal;
        case P::Chelated:          return InteractionKey::Chelated;
        default:                   return InteractionKey::Unknown;
    }
}

double point_distance(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    return std::sqrt(
        interaction::distance2(a, b)
    );
}

bool points_are_aligned(
    const interaction::point3D& a,
    const interaction::point3D& b,
    const interaction::point3D& c)
{
    const double abx = b.x - a.x;
    const double aby = b.y - a.y;
    const double abz = b.z - a.z;

    const double acx = c.x - a.x;
    const double acy = c.y - a.y;
    const double acz = c.z - a.z;

    const double cross_x = aby * acz - abz * acy;
    const double cross_y = abz * acx - abx * acz;
    const double cross_z = abx * acy - aby * acx;

    const double cross_norm2 =
        cross_x * cross_x +
        cross_y * cross_y +
        cross_z * cross_z;

    return cross_norm2 < 1e-6;
}

} // namespace

std::vector<Triplet> generate_triplets(
    const std::vector<interaction::interaction_area>& areas,
    double max_distance,
    AreaRole role)
{
    std::vector<Triplet> triplets;

    for (std::size_t i = 0; i < areas.size(); ++i) {

        for (std::size_t j = i + 1; j < areas.size(); ++j) {

            const double dij =
                point_distance(
                    areas[i].centroid,
                    areas[j].centroid
                );

            if (dij > max_distance || dij < 0.001)
                continue;

            for (std::size_t k = j + 1; k < areas.size(); ++k) {

                const double dik =
                    point_distance(
                        areas[i].centroid,
                        areas[k].centroid
                    );

                const double djk =
                    point_distance(
                        areas[j].centroid,
                        areas[k].centroid
                    );

                if (dik > max_distance || djk > max_distance)
                    continue;

                if (dik < 0.001 || djk < 0.001)
                    continue;

                if (points_are_aligned(
                        areas[i].centroid,
                        areas[j].centroid,
                        areas[k].centroid))
                    continue;

                const auto to_key =
                    (role == AreaRole::Ligand)
                        ? to_ligand_key
                        : to_protein_key;

                triplets.push_back({
                    {i, j, k},
                    {
                        to_key(
                            areas[i].pattern,
                            areas[i].pattern_class
                        ),
                        to_key(
                            areas[j].pattern,
                            areas[j].pattern_class
                        ),
                        to_key(
                            areas[k].pattern,
                            areas[k].pattern_class
                        )
                    },
                    {dij, dik, djk}
                });
            }
        }
    }

    return triplets;
}