#include "helpers.hpp"

#include <cmath>

void differentiation(std::vector<double> points, double dx, std::size_t iter)
{
    std::vector<double> result;

    for (std::size_t it = 0; it < iter; ++it)
    {
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            result.emplace_back((points[i+1] - points[i]) / dx);
        }
    study::do_not_optimize(result);
    }
}

void differentiation_simd(std::vector<double> points, double dx, std::size_t iter)
{
    std::vector<double> result;

    #pragma omp simd
    for (std::size_t it = 0; it < iter; ++it)
    {
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            result.emplace_back((points[i+1] - points[i]) / dx);
        }
    study::do_not_optimize(result);
    }
}

void differentiation_parallel(std::vector<double> points, double dx, std::size_t iter)
{
    std::vector<double> result;

    #pragma omp parallel for
    for (std::size_t it = 0; it < iter; ++it)
    {
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            result.emplace_back((points[i+1] - points[i]) / dx);
        }
    study::do_not_optimize(result);
    }
}

int main()
{
    std::vector<study::result_tuple> result_vector;
    double dx = 283747382.;
    std::vector<double> points = {3., 5, 101093., 3984923.};

    for (std::uint64_t iter_pow = 3;  iter_pow < 6; ++iter_pow)
    {
        std::uint64_t iter = std::pow(10, iter_pow);
        result_vector.emplace_back(study::benchmark_math_function(differentiation, points, dx, iter, "no optimization"));
        result_vector.emplace_back(study::benchmark_math_function(differentiation_simd, points, dx, iter, "simd optimization"));
        result_vector.emplace_back(study::benchmark_math_function(differentiation_parallel, points, dx, iter, "parallel optimization"));
    }

    study::statistics_to_csv("statistics", result_vector);
    return 0;
}