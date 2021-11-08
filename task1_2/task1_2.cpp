#include "helpers.hpp"

#include <cmath>

void differentiation(const std::vector<double>& points, double dx, std::vector<double>& result)
{
    for (std::size_t i = 0; i < points.size() - 1; ++i)
        result[i] = (points[i+1] - points[i]) / dx;
}

void differentiation_simd(const std::vector<double>& points, double dx, std::vector<double>& result)
{
    #pragma omp simd
    for (std::size_t i = 0; i < points.size() - 1; ++i)
        result[i] = (points[i+1] - points[i]) / dx;
}

void differentiation_parallel(const std::vector<double>& points, double dx, std::vector<double>& result)
{
    #pragma omp parallel for num_threads(3)
    for (std::size_t i = 0; i < points.size() - 1; ++i)
        result[i] = (points[i+1] - points[i]) / dx;
}

double func(double x)
{
    return std::pow(x-5, 4);
}

std::tuple<std::vector<double>, double> generating_points(std::function<double(double)> func, double from, double to, std::size_t num_points)
{
    std::vector<double> points;
    points.reserve(num_points);

    double dx = (to - from) / num_points;

    for (double point = from; point <= to; point += dx)
        points.emplace_back(func(point));

    return {points, dx};
}

int main()
{
    std::vector<study::result_tuple> result_vector;
    auto [points, dx] = generating_points(func, 100, 545, 100000);

    for (std::uint64_t iter_pow = 3;  iter_pow < 6; ++iter_pow)
    {
        std::uint64_t iter = std::pow(10, iter_pow);
        result_vector.emplace_back(study::benchmark_math_function(differentiation, points, dx, iter, "no pragma"));
        result_vector.emplace_back(study::benchmark_math_function(differentiation_simd, points, dx, iter, "simd"));
        result_vector.emplace_back(study::benchmark_math_function(differentiation_parallel, points, dx, iter, "parallel"));
    }

    study::statistics_to_csv("statistics", result_vector);
    return 0;
}
