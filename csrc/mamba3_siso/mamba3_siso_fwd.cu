// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "mamba3_siso.h"
#include "../utils/cuda_common.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

__device__ __forceinline__ float sigmoidf_fast(float x) {
	return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float siluf_fast(float x) {
	return x * sigmoidf_fast(x);
}

__device__ __forceinline__ float wrap_two_pi(float x) {
	return x - kTwoPi * floorf(x / kTwoPi);
}

__device__ __forceinline__ int q_index(const Mamba3SisoParams params, int b, int t, int g, int n) {
	return (((b * params.seqlen + t) * params.nheads_qk + g) * params.headdim_qk + n);
}

__device__ __forceinline__ int v_index(const Mamba3SisoParams params, int b, int t, int h, int p) {
	return (((b * params.seqlen + t) * params.nheads + h) * params.headdim_v + p);
}

__device__ __forceinline__ int bhs_index(const Mamba3SisoParams params, int b, int h, int t) {
	return ((b * params.nheads + h) * params.seqlen + t);
}

__device__ __forceinline__ int bias_index(const Mamba3SisoParams params, int h, int n) {
	return h * params.headdim_qk + n;
}

__device__ __forceinline__ int angle_index(const Mamba3SisoParams params, int b, int t, int h, int a) {
	return (((b * params.seqlen + t) * params.nheads + h) * params.headdim_angles + a);
}

__global__ void mamba3_siso_fwd_kernel(Mamba3SisoParams params) {
	const int h = blockIdx.x;
	const int b = blockIdx.y;
	const int tid = threadIdx.x;
	const int threads = blockDim.x;

	const int n = params.headdim_qk;
	const int p_dim = params.headdim_v;
	const int angle_dim = params.headdim_angles;
	const int qk_heads_per_group = params.nheads / params.nheads_qk;
	const int g = h / qk_heads_per_group;

	extern __shared__ float smem[];
	float *state = smem;                                  // [P, N]
	float *angle_state = state + p_dim * n;               // [A]
	float *q_pre = angle_state + angle_dim;               // [N]
	float *k_pre = q_pre + n;                             // [N]
	float *q_rot = k_pre + n;                             // [N]
	float *k_scaled = q_rot + n;                          // [N]
	float *scratch = k_scaled + n;                        // [threads]

	for (int i = tid; i < p_dim * n; i += threads) {
		state[i] = 0.0f;
	}
	for (int i = tid; i < angle_dim; i += threads) {
		angle_state[i] = 0.0f;
	}
	__syncthreads();

	for (int t = 0; t < params.seqlen; ++t) {
		const float dt = params.dt_ptr[bhs_index(params, b, h, t)];

		for (int a = tid; a < angle_dim; a += threads) {
			const float raw_angle = params.angles_ptr[angle_index(params, b, t, h, a)];
			const float next_angle = angle_state[a] + tanhf(raw_angle) * dt * kPi;
			angle_state[a] = wrap_two_pi(next_angle);
		}
		__syncthreads();

		float qk_partial = 0.0f;
		for (int i = tid; i < n; i += threads) {
			const float q_val = params.q_ptr[q_index(params, b, t, g, i)] + params.q_bias_ptr[bias_index(params, h, i)];
			const float k_val = params.k_ptr[q_index(params, b, t, g, i)] + params.k_bias_ptr[bias_index(params, h, i)];
			q_pre[i] = q_val;
			k_pre[i] = k_val;
			qk_partial += q_val * k_val;
		}
		scratch[tid] = qk_partial;
		__syncthreads();

		for (int offset = threads / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				scratch[tid] += scratch[tid + offset];
			}
			__syncthreads();
		}
		const float qk_dot = scratch[0];

		for (int pair = tid; pair < n / 2; pair += threads) {
			const float theta = pair < angle_dim ? angle_state[pair] : 0.0f;
			const float c = cosf(theta);
			const float s = sinf(theta);
			const int i0 = 2 * pair;
			const int i1 = i0 + 1;

			const float q0 = q_pre[i0];
			const float q1 = q_pre[i1];
			const float k0 = k_pre[i0];
			const float k1 = k_pre[i1];
			q_rot[i0] = q0 * c - q1 * s;
			q_rot[i1] = q0 * s + q1 * c;
			k_pre[i0] = k0 * c - k1 * s;
			k_pre[i1] = k0 * s + k1 * c;
		}
		__syncthreads();

		const float trap = params.trap_ptr[bhs_index(params, b, h, t)];
		const float gamma = dt * sigmoidf_fast(trap);
		float shifted_gamma = 0.0f;
		if (t + 1 < params.seqlen) {
			const float next_dt = params.dt_ptr[bhs_index(params, b, h, t + 1)];
			const float next_trap = params.trap_ptr[bhs_index(params, b, h, t + 1)];
			shifted_gamma = next_dt * (1.0f - sigmoidf_fast(next_trap));
		}
		const float scale = gamma + shifted_gamma;

		for (int i = tid; i < n; i += threads) {
			k_scaled[i] = k_pre[i] * scale;
		}

		const float decay = expf(params.adt_ptr[bhs_index(params, b, h, t)]);
		for (int i = tid; i < p_dim * n; i += threads) {
			state[i] *= decay;
		}
		__syncthreads();

		const float d_val = params.has_d ? params.d_ptr[h] : 0.0f;
		const float diagonal_scale = d_val + qk_dot * gamma;
		for (int p = tid; p < p_dim; p += threads) {
			float acc = 0.0f;
			const float *state_row = state + p * n;
			for (int i = 0; i < n; ++i) {
				acc += q_rot[i] * state_row[i];
			}
			const float v_val = params.v_ptr[v_index(params, b, t, h, p)];
			float out = acc + diagonal_scale * v_val;
			if (params.has_z) {
				const float z_val = params.z_ptr[v_index(params, b, t, h, p)];
				out *= siluf_fast(z_val);
			}
			params.out_ptr[v_index(params, b, t, h, p)] = out;
		}
		__syncthreads();

		for (int i = tid; i < p_dim * n; i += threads) {
			const int p = i / n;
			const int qk_dim = i - p * n;
			const float v_val = params.v_ptr[v_index(params, b, t, h, p)];
			state[i] += v_val * k_scaled[qk_dim];
		}
		__syncthreads();
	}
}

}  // namespace

void mamba3_siso_fwd_cuda(Mamba3SisoParams &params, cudaStream_t stream) {
	if (params.nheads_qk <= 0 || params.nheads % params.nheads_qk != 0) {
		throw std::runtime_error("Mamba3SisoFn requires nheads divisible by nheads_qk");
	}
	if (params.headdim_qk <= 0 || params.headdim_v <= 0 || params.headdim_angles <= 0) {
		throw std::runtime_error("Mamba3SisoFn received invalid head dimensions");
	}
	if (params.headdim_qk % 2 != 0) {
		throw std::runtime_error("Mamba3SisoFn requires even headdim_qk for rotary embeddings");
	}
	if (params.headdim_angles > params.headdim_qk / 2) {
		throw std::runtime_error("Mamba3SisoFn requires headdim_angles <= headdim_qk / 2");
	}

	const int threads = 128;
	const size_t shared_floats =
		static_cast<size_t>(params.headdim_v) * params.headdim_qk +
		static_cast<size_t>(params.headdim_angles) +
		static_cast<size_t>(4) * params.headdim_qk +
		static_cast<size_t>(threads);
	const size_t shared_bytes = shared_floats * sizeof(float);
	int device = 0;
	int max_dynamic_shared_bytes = 0;
	XM_CUDA_CHECK(cudaGetDevice(&device));
	XM_CUDA_CHECK(cudaDeviceGetAttribute(&max_dynamic_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
	if (shared_bytes > static_cast<size_t>(max_dynamic_shared_bytes)) {
		throw std::runtime_error("Mamba3SisoFn shared-memory requirement exceeds this GPU limit");
	}

	XM_CUDA_CHECK(cudaFuncSetAttribute(mamba3_siso_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
	const dim3 grid(params.nheads, params.batch);
	mamba3_siso_fwd_kernel<<<grid, threads, shared_bytes, stream>>>(params);
	XM_CUDA_KERNEL_CHECK();
}
