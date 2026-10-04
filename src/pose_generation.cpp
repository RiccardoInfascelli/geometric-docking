#include "pose_generation.hpp"
#include <Eigen/Dense>

#include <cmath>

namespace
{

struct Vec3
{
    double x{};
    double y{};
    double z{};
};


Vec3 subtract(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    return {
        static_cast<double>(a.x) -
            static_cast<double>(b.x),

        static_cast<double>(a.y) -
            static_cast<double>(b.y),

        static_cast<double>(a.z) -
            static_cast<double>(b.z)
    };
}


double norm(const Vec3& v)
{
    return std::sqrt(
        v.x * v.x +
        v.y * v.y +
        v.z * v.z
    );
}


Vec3 normalize(const Vec3& v)
{
    const double n = norm(v);

    return {
        v.x / n,
        v.y / n,
        v.z / n
    };
}


Vec3 cross(
    const Vec3& a,
    const Vec3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}


bool build_frame(
    const std::array<interaction::point3D, 3>& points,
    std::array<Vec3, 3>& frame)
{
    constexpr double epsilon = 1e-10;

    const Vec3 v1 =
        subtract(
            points[1],
            points[0]
        );

    const Vec3 v2 =
        subtract(
            points[2],
            points[0]
        );

    if (norm(v1) < epsilon)
        return false;

    const Vec3 e1 =
        normalize(v1);

    const Vec3 normal =
        cross(e1, v2);

    if (norm(normal) < epsilon)
        return false;

    const Vec3 e3 =
        normalize(normal);

    const Vec3 e2 =
        cross(e3, e1);

    frame = {
        e1,
        e2,
        e3
    };

    return true;
}


double component(
    const Vec3& v,
    int index)
{
    if (index == 0)
        return v.x;

    if (index == 1)
        return v.y;

    return v.z;
}

} // namespace


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