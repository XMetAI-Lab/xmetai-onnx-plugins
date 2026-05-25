#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#ifndef USE_ROCM
    #include <cuda_bf16.h>
#else
    #include <hip/hip_bf16.h>
#endif
#include <cuda_fp16.h>

namespace xmetai::cuda {

inline void Check(cudaError_t err, const char *expr, const char *file, int line) {
    if (err == cudaSuccess) {
        return;
    }
    std::ostringstream oss;
    oss << "CUDA call failed: " << expr << " returned " << cudaGetErrorString(err)
        << " at " << file << ":" << line;
    throw std::runtime_error(oss.str());
}

inline void KernelCheck(const char *file, int line) {
    auto err = cudaGetLastError();
    if (err == cudaSuccess) {
        return;
    }
    std::ostringstream oss;
    oss << "CUDA kernel launch failed: " << cudaGetErrorString(err)
        << " at " << file << ":" << line;
    throw std::runtime_error(oss.str());
}

using half = __half;
#ifdef USE_ROCM
using bfloat16 = __hip_bfloat16;
#else
using bfloat16 = __nv_bfloat16;
#endif

struct Complex {
    float real_;
    float imag_;
    __host__ __device__ constexpr Complex(float real = 0.f, float imag = 0.f) : real_(real), imag_(imag) {}
};

__host__ __device__ inline constexpr Complex operator+(const Complex &a, const Complex &b) {
    return Complex(a.real_ + b.real_, a.imag_ + b.imag_);
}

__host__ __device__ inline constexpr Complex operator*(const Complex &a, const Complex &b) {
    return Complex(a.real_ * b.real_ - a.imag_ * b.imag_, a.real_ * b.imag_ + a.imag_ * b.real_);
}

__host__ __device__ inline constexpr Complex operator*(const Complex &a, float s) {
    return Complex(a.real_ * s, a.imag_ * s);
}

__host__ __device__ inline constexpr Complex operator*(float s, const Complex &a) {
    return Complex(a.real_ * s, a.imag_ * s);
}

__host__ __device__ inline Complex &operator*=(Complex &a, float s) {
    a.real_ *= s;
    a.imag_ *= s;
    return a;
}

}  // namespace xmetai::cuda

#define XM_CUDA_CHECK(expr) ::xmetai::cuda::Check((expr), #expr, __FILE__, __LINE__)
#define XM_CUDA_KERNEL_CHECK() ::xmetai::cuda::KernelCheck(__FILE__, __LINE__)
