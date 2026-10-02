#include <Eigen/Core>
#include <GraphMol/ROMol.h>

#include <interaction/elements.hpp>

#include <iostream>

int main()
{
    const Eigen::Vector3d point{1.0, 2.0, 3.0};

    const auto carbon =
        interaction::parse_chemical_element("C");

    if (!carbon.has_value()) {
        std::cerr << "Cannot parse carbon element\n";
        return 1;
    }

    std::cout
        << "Environment OK\n"
        << "Eigen point: "
        << point.transpose()
        << "\n"
        << "RDKit ROMol type size: "
        << sizeof(RDKit::ROMol)
        << "\n";

    return 0;
}