#include "pose_generation.hpp"



#include "clash.hpp"



#include <Eigen/Dense>



#include <algorithm>

#include <array>

#include <chrono>

#include <cmath>





bool rigid_transform_from_points(

    const std::array<interaction::point3D, 3>& source,

    const std::array<interaction::point3D, 3>& target,

    RigidTransform& transform)

{

    Eigen::Matrix<double, 3, 3> P;

    Eigen::Matrix<double, 3, 3> Q;



    for (int i = 0; i < 3; ++i) {



        P.col(i) = Eigen::Vector3d(

            source[i].x,

            source[i].y,

            source[i].z

        );



        Q.col(i) = Eigen::Vector3d(

            target[i].x,

            target[i].y,

            target[i].z

        );

    }



    const Eigen::Vector3d p_centroid =

        P.rowwise().mean();



    const Eigen::Vector3d q_centroid =

        Q.rowwise().mean();



    const Eigen::Matrix3d Pc =

        P.colwise() - p_centroid;



    const Eigen::Matrix3d Qc =

        Q.colwise() - q_centroid;



    const Eigen::Matrix3d H =

        Pc * Qc.transpose();



    Eigen::JacobiSVD<Eigen::Matrix3d> svd(

        H,

        Eigen::ComputeFullU |

        Eigen::ComputeFullV

    );



    Eigen::Matrix3d U =

        svd.matrixU();



    Eigen::Matrix3d V =

        svd.matrixV();



    Eigen::Matrix3d D =

        Eigen::Matrix3d::Identity();



    if (

        (V * U.transpose()).determinant()

        < 0.0

    ) {

        D(2, 2) = -1.0;

    }



    const Eigen::Matrix3d R =

        V * D * U.transpose();



    const Eigen::Vector3d t =

        q_centroid -

        R * p_centroid;





    for (int row = 0; row < 3; ++row) {

        for (int col = 0; col < 3; ++col) {



            transform.rotation[

                row * 3 + col

            ] = R(row, col);

        }

    }



    transform.translation = {

        t.x(),

        t.y(),

        t.z()

    };



    return true;

}



namespace

{



double rmsd_from_transform(

    const RDKit::ROMol& ligand,

    const RigidTransform& transform)

{

    const auto& conformer =

        ligand.getConformer();



    double squared_sum = 0.0;



    for (

        unsigned int atom_index = 0;

        atom_index < ligand.getNumAtoms();

        ++atom_index

    ) {

        const auto position =

            conformer.getAtomPos(

                atom_index

            );



        const interaction::point3D original{

            static_cast<interaction::coordinate_type>(

                position.x

            ),

            static_cast<interaction::coordinate_type>(

                position.y

            ),

            static_cast<interaction::coordinate_type>(

                position.z

            )

        };



        const auto transformed =

            transform_point(

                original,

                transform

            );



        const double dx =

            static_cast<double>(

                transformed.x

            ) - position.x;



        const double dy =

            static_cast<double>(

                transformed.y

            ) - position.y;



        const double dz =

            static_cast<double>(

                transformed.z

            ) - position.z;



        squared_sum +=

            dx * dx +

            dy * dy +

            dz * dz;

    }



    return std::sqrt(

        squared_sum /

        static_cast<double>(

            ligand.getNumAtoms()

        )

    );

}



} // namespace





PoseGenerationResult generate_candidate_poses(

    const RDKit::ROMol& ligand,

    const std::vector<MatchingArea>& ligand_matching_areas,

    const std::vector<Triplet>& ligand_triplets,

    const std::vector<PoseTarget>& pose_targets,

    const ProteinGrid& protein_grid,

    double ligand_diameter,

    double triplet_bin_size,

    double operational_clash_threshold,

    const std::array<double, 4>& clash_thresholds)

{

    PoseGenerationResult result;



    result.zero_clash_grid_histogram.assign(

        ligand_matching_areas.size() + 1,

        0

    );



    using Clock =

        std::chrono::steady_clock;





    // ========================================================

    // Protein target triplet hash

    // ========================================================





    const auto index_start =

        Clock::now();





    const auto protein_triplet_hash =

        build_protein_triplet_hash_table(

            pose_targets,

            ligand_diameter,

            triplet_bin_size

        );





    const auto index_end =

        Clock::now();





    result.timings

        .protein_triplet_index_seconds =

            std::chrono::duration<double>(

                index_end -

                index_start

            ).count();





    for (const auto& [key, bucket] :

         protein_triplet_hash) {



        (void)key;



        result.protein_triplet_records +=

            bucket.size();

    }



    result.protein_hash_keys =

        protein_triplet_hash.size();





    // ========================================================

    // Candidate pose generation

    // ========================================================



    double grid_clash_seconds = 0.0;

    double oracle_rmsd_seconds = 0.0;


    const auto generation_start =

        Clock::now();





    for (const auto& ligand_triplet :

         ligand_triplets) {



        const auto canonical =

            canonicalize_triplet(

                ligand_triplet,

                triplet_bin_size

            );



        const auto buckets =

            find_neighbor_triplet_buckets(

                protein_triplet_hash,

                canonical

            );


        if (buckets.empty())

            continue;


        ++result.ligand_triplets_with_match;


        for (const auto* bucket :

             buckets) {

            result.total_candidate_matches +=

                bucket->size();


            result.largest_bucket =

                std::max(

                    result.largest_bucket,

                    bucket->size()

                );

        }





        for (const auto* bucket :

             buckets) {



            for (const auto& protein_record :

                 *bucket) {



                const std::array<

                    interaction::point3D,

                    3

                > target_points{



                    pose_targets[

                        protein_record.indices[0]

                    ].position,



                    pose_targets[

                        protein_record.indices[1]

                    ].position,



                    pose_targets[

                        protein_record.indices[2]

                    ].position

                };





                for (

                    std::size_t order_index = 0;

                    order_index <

                        canonical.equivalent_order_count;

                    ++order_index

                ) {

                    const auto& ligand_order =

                        canonical.equivalent_orders[

                            order_index

                        ];





                    const std::array<

                        interaction::point3D,

                        3

                    > source_points{



                        ligand_matching_areas[

                            ligand_order[0]

                        ].centroid,



                        ligand_matching_areas[

                            ligand_order[1]

                        ].centroid,



                        ligand_matching_areas[

                            ligand_order[2]

                        ].centroid

                    };





                    ++result.transforms_generated;





                    RigidTransform transform;





                    if (

                        !rigid_transform_from_points(

                            source_points,

                            target_points,

                            transform

                        )

                    ) {

                        continue;

                    }





                    ++result.transforms_valid;





                    // ----------------------------------------

                    // Fast grid score

                    // ----------------------------------------



                    const auto grid_clash_start =

                        Clock::now();


                    const auto transformed_matching =

                        transform_matching_areas(

                            ligand_matching_areas,

                            transform

                        );



                    const auto grid_score =

                        score_ligand_on_grid(

                            transformed_matching,

                            protein_grid

                        );





                    if (

                        grid_score.compatible_areas > 0

                    ) {

                        ++result.poses_with_grid_support;

                    }





                    result.best_grid_score =

                        std::max(

                            result.best_grid_score,

                            grid_score.compatible_areas

                        );





                    // ----------------------------------------

                    // Fast clash

                    // ----------------------------------------



                    const auto clash =

                        score_fast_grid_clash(

                            ligand,

                            transform,

                            protein_grid

                        );


                    const auto grid_clash_end =

                        Clock::now();


                    grid_clash_seconds +=

                        std::chrono::duration<double>(

                            grid_clash_end -

                            grid_clash_start

                        ).count();





                    if (

                        clash.fraction >

                        clash_thresholds.back()

                    ) {

                        ++result.poses_over_five_clashes;

                        continue;

                    }





                    // ORACLE ONLY

                    const auto rmsd_start =

                        Clock::now();


                    const double rmsd =

                        rmsd_from_transform(

                            ligand,

                            transform

                        );


                    const auto rmsd_end =

                        Clock::now();


                    oracle_rmsd_seconds +=

                        std::chrono::duration<double>(

                            rmsd_end -

                            rmsd_start

                        ).count();





                    // ----------------------------------------

                    // Clash threshold experiment

                    // ----------------------------------------



                    for (

                        std::size_t threshold_index = 0;

                        threshold_index <

                            clash_thresholds.size();

                        ++threshold_index

                    ) {

                        if (

                            clash.fraction <=

                            clash_thresholds[

                                threshold_index

                            ]

                        ) {

                            ++result

                                .poses_below_clash_threshold[

                                    threshold_index

                                ];



                            result

                                .best_rmsd_below_clash_threshold[

                                    threshold_index

                                ] =

                                    std::min(

                                        result

                                            .best_rmsd_below_clash_threshold[

                                                threshold_index

                                            ],

                                        rmsd

                                    );

                        }

                    }





                    // ----------------------------------------

                    // Absolute clash diagnostic

                    // ----------------------------------------



                    if (clash.occupied_atoms == 0) {



                        ++result.poses_zero_clash;



                        result.best_zero_clash_rmsd =

                            std::min(

                                result.best_zero_clash_rmsd,

                                rmsd

                            );



                        result.best_grid_score_zero_clash =

                            std::max(

                                result.best_grid_score_zero_clash,

                                grid_score.compatible_areas

                            );



                        ++result

                            .zero_clash_grid_histogram[

                                grid_score.compatible_areas

                            ];

                    }

                    else if (clash.occupied_atoms == 1) {



                        ++result.poses_one_clash;

                    }

                    else if (clash.occupied_atoms == 2) {



                        ++result.poses_two_clashes;

                    }

                    else if (clash.occupied_atoms <= 5) {



                        ++result

                            .poses_three_to_five_clashes;

                    }

                    else {



                        ++result.poses_over_five_clashes;

                    }





                    // ----------------------------------------

                    // Operational 5% survivor set

                    // ----------------------------------------



                    if (

                        clash.fraction >

                        operational_clash_threshold

                    ) {

                        continue;

                    }





                    RankedPoseDiagnostic record;



                    record.compatible_areas =

                        grid_score.compatible_areas;



                    record.occupied_atoms =

                        clash.occupied_atoms;



                    record.clash_fraction =

                        clash.fraction;



                    record.rmsd =

                        rmsd;



                    record.transform =

                        transform;





                    result.surviving_poses.push_back(

                        record

                    );

                }

            }

        }

    }





    const auto generation_end =

        Clock::now();





    result.generation_seconds =

        std::chrono::duration<double>(

            generation_end -

            generation_start

        ).count();


    result.timings.grid_clash_seconds =

        grid_clash_seconds;


    result.timings.oracle_rmsd_seconds =

        oracle_rmsd_seconds;


    const double matching_kabsch_seconds =

        result.generation_seconds -

        grid_clash_seconds -

        oracle_rmsd_seconds;


    result.timings.matching_kabsch_seconds =

        std::max(

            0.0,

            matching_kabsch_seconds

        );





    return result;

}