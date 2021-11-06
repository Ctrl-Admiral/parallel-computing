#ifndef PARALLEL_COMPUTING_HELPERS_HPP_
#define PARALLEL_COMPUTING_HELPERS_HPP_

#include <cstdint>
#include <tuple>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <iostream>
#include <fstream>

namespace study
{
using namespace std::chrono;

using result_tuple = std::tuple<std::string /*operation name*/, std::uint64_t /*iterations*/, double /*ticks*/, double /*nanoseconds*/>;

template <typename T>
inline __attribute__((always_inline)) void do_not_optimize(T& value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

inline __attribute__((always_inline)) std::uint64_t ticks()
{
    std::uint64_t tsc;
    asm volatile("mfence; "         // memory barrier
                 "rdtsc; "          // read of tsc
                 "shl $32,%%rdx; "  // shift higher 32 bits stored in rdx up
                 "or %%rdx,%%rax"   // and or onto rax
                 : "=a"(tsc)        // output to tsc
                 :
                 : "%rcx", "%rdx", "memory");
    return tsc;
}

template<typename NumType, typename FuncType>
result_tuple benchmark_math_function(const FuncType& func, NumType num, std::size_t iter, const std::string& func_name)
{
    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    std::uint64_t start_ticks = study::ticks();

    func(num, iter);

    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    std::uint64_t end_ticks = study::ticks();

    std::uint64_t ns = static_cast<std::uint64_t>(duration_cast<nanoseconds>(end_time - start_time).count());
    std::uint64_t ticks = end_ticks - start_ticks;

    return std::make_tuple(func_name, iter, ticks, ns);
}

void statistics_to_csv(const std::string& filename, const std::vector<result_tuple>& statistics, char sep=',')
{
    std::ofstream f(filename);
    if (!f.is_open()) throw std::runtime_error("Cannot open output csv-file");

    for (const auto& elem : statistics)
    {
        const auto& [ oper_name, iter, ticks, ns] = elem;
        f << oper_name << sep << iter << sep << ticks << sep << ns << '\n';
    }
    f.close();
}


} // study

#endif // PARALLEL_COMPUTING_HELPERS_HPP_
