#include "helpers.hpp"
#include <cmath>
#include <immintrin.h>  // For AVX2


__m256d double_to_m256d(double num_scalar)
{
    // there are 4 doubles in 256-bit register
    const double array[4] = {num_scalar, num_scalar, num_scalar, num_scalar};
    return _mm256_load_pd(&array[0]);
}

void independent_scalar_sqrt(std::int64_t num, std::size_t iter)
{
    for (std::size_t i = 0; i < iter; ++i)
    {
        double sqrt = std::sqrt(num);
        study::do_not_optimize(sqrt);
    }
}

void dependent_scalar_sqrt(std::int64_t num, std::size_t iter)
{
    for (std::size_t i = 0; i < iter; ++i)
    {
        num = std::sqrt(num);
    }
    study::do_not_optimize(num);
}

void independent_vector_sqrt(__m256d& num, std::size_t iter_as_for_scalar)
{
    for (std::size_t i = 0; i < iter_as_for_scalar / 4; ++i)
    {
        __m256d sqrt = _mm256_sqrt_pd(num);
        study::do_not_optimize(sqrt);
    }
}

void dependent_vector_sqrt(__m256d& num, std::size_t iter_as_for_scalar)
{
    for (std::size_t i = 0; i < iter_as_for_scalar / 4; ++i)
    {
        num = _mm256_sqrt_pd(num);
    }
    study::do_not_optimize(num);
}


int main()
{
    std::vector<study::result_tuple> result_vector;
    double num = 283747382.;
    __m256d num_vec = double_to_m256d(num);

    for (std::uint64_t iter_pow = 3;  iter_pow < 9; ++iter_pow)
    {
        std::uint64_t iter = std::pow(10, iter_pow);

        result_vector.emplace_back(study::benchmark_math_function(independent_scalar_sqrt, num, iter, "Independent scalar sqrt"));
        result_vector.emplace_back(study::benchmark_math_function(dependent_scalar_sqrt, num, iter, "Dependent scalar sqrt"));

        result_vector.emplace_back(study::benchmark_math_function(independent_vector_sqrt, num_vec, iter, "Independent vector sqrt"));
        result_vector.emplace_back(study::benchmark_math_function(dependent_vector_sqrt, num_vec, iter, "Dependent vector sqrt"));
    }

    study::statistics_to_csv("statistics", result_vector);

    return 0;
}
