// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "gated_deltanet2.h"
#include "../utils/cuda_common.h"

namespace {

constexpr float kL2NormEps = 1.0e-6f;

__device__ __forceinline__ int qk_index(const GatedDeltaNet2Params params, int b, int t, int h, int d) {
	return (((b * params.seqlen + t) * params.nheads + h) * params.headdim_k + d);
}

__device__ __forceinline__ int vv_index(const GatedDeltaNet2Params params, int b, int t, int h, int d) {
	return (((b * params.seqlen + t) * params.nheads + h) * params.headdim_v + d);
}

__device__ __forceinline__ int state_index(const GatedDeltaNet2Params params, int b, int h, int kd, int vd) {
	return (((b * params.nheads + h) * params.headdim_k + kd) * params.headdim_v + vd);
}

__global__ void gated_deltanet2_fwd_kernel(GatedDeltaNet2Params params) {
	const int bh = blockIdx.x;
	const int h = bh % params.nheads;
	const int b_idx = bh / params.nheads;
	const int tid = threadIdx.x;
	const int threads = blockDim.x;
	const int k_dim = params.headdim_k;
	const int v_dim = params.headdim_v;
	const int state_size = k_dim * v_dim;

	extern __shared__ float smem[];
	float *q_vec = smem;                    // [K]
	float *k_vec = q_vec + k_dim;           // [K]
	float *v_new = k_vec + k_dim;           // [V]
	float *scratch_q = v_new + v_dim;       // [threads]
	float *scratch_k = scratch_q + threads; // [threads]

	for (int i = tid; i < state_size; i += threads) {
		const int kd = i / v_dim;
		const int vd = i - kd * v_dim;
		const int idx = state_index(params, b_idx, h, kd, vd);
		params.state_ptr[idx] = params.has_initial_state ? params.initial_state_ptr[idx] : 0.0f;
	}
	__syncthreads();

	for (int t = 0; t < params.seqlen; ++t) {
		float q_sum = 0.0f;
		float k_sum = 0.0f;
		for (int d = tid; d < k_dim; d += threads) {
			const float q_val = params.q_ptr[qk_index(params, b_idx, t, h, d)];
			const float k_val = params.k_ptr[qk_index(params, b_idx, t, h, d)];
			q_sum += q_val * q_val;
			k_sum += k_val * k_val;
		}
		scratch_q[tid] = q_sum;
		scratch_k[tid] = k_sum;
		__syncthreads();

		for (int offset = threads / 2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				scratch_q[tid] += scratch_q[tid + offset];
				scratch_k[tid] += scratch_k[tid + offset];
			}
			__syncthreads();
		}

		const float q_mul = params.use_qk_l2norm_in_kernel ? rsqrtf(scratch_q[0] + kL2NormEps) * params.scale : params.scale;
		const float k_mul = params.use_qk_l2norm_in_kernel ? rsqrtf(scratch_k[0] + kL2NormEps) : 1.0f;
		for (int d = tid; d < k_dim; d += threads) {
			q_vec[d] = params.q_ptr[qk_index(params, b_idx, t, h, d)] * q_mul;
			k_vec[d] = params.k_ptr[qk_index(params, b_idx, t, h, d)] * k_mul;
		}
		__syncthreads();

		for (int i = tid; i < state_size; i += threads) {
			const int kd = i / v_dim;
			const int idx = state_index(params, b_idx, h, kd, i - kd * v_dim);
			params.state_ptr[idx] *= expf(params.g_ptr[qk_index(params, b_idx, t, h, kd)]);
		}
		__syncthreads();

		for (int vd = tid; vd < v_dim; vd += threads) {
			float erase = 0.0f;
			for (int kd = 0; kd < k_dim; ++kd) {
				const int s_idx = state_index(params, b_idx, h, kd, vd);
				const float beta_k = params.b_ptr[qk_index(params, b_idx, t, h, kd)] * k_vec[kd];
				erase += beta_k * params.state_ptr[s_idx];
			}
			const float value = params.v_ptr[vv_index(params, b_idx, t, h, vd)];
			const float write_gate = params.w_ptr[vv_index(params, b_idx, t, h, vd)];
			v_new[vd] = write_gate * value - erase;
		}
		__syncthreads();

		for (int i = tid; i < state_size; i += threads) {
			const int kd = i / v_dim;
			const int vd = i - kd * v_dim;
			const int s_idx = state_index(params, b_idx, h, kd, vd);
			params.state_ptr[s_idx] += k_vec[kd] * v_new[vd];
		}
		__syncthreads();

		for (int vd = tid; vd < v_dim; vd += threads) {
			float acc = 0.0f;
			for (int kd = 0; kd < k_dim; ++kd) {
				acc += q_vec[kd] * params.state_ptr[state_index(params, b_idx, h, kd, vd)];
			}
			params.out_ptr[vv_index(params, b_idx, t, h, vd)] = acc;
		}
		__syncthreads();
	}
}

}  // namespace

void gated_deltanet2_fwd_cuda(GatedDeltaNet2Params &params, cudaStream_t stream) {
	if (params.batch <= 0 || params.seqlen <= 0 || params.nheads <= 0 || params.headdim_k <= 0 || params.headdim_v <= 0) {
		throw std::runtime_error("GatedDeltaNet2Fn received invalid tensor dimensions");
	}
	const int threads = 256;
	const size_t shared_floats =
		static_cast<size_t>(2 * params.headdim_k) +
		static_cast<size_t>(params.headdim_v) +
		static_cast<size_t>(2 * threads);
	const size_t shared_bytes = shared_floats * sizeof(float);
	int device = 0;
	int max_dynamic_shared_bytes = 0;
	XM_CUDA_CHECK(cudaGetDevice(&device));
	XM_CUDA_CHECK(cudaDeviceGetAttribute(&max_dynamic_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
	if (shared_bytes > static_cast<size_t>(max_dynamic_shared_bytes)) {
		throw std::runtime_error("GatedDeltaNet2Fn shared-memory requirement exceeds this GPU limit");
	}

	XM_CUDA_CHECK(cudaFuncSetAttribute(gated_deltanet2_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
	const dim3 grid(params.batch * params.nheads);
	gated_deltanet2_fwd_kernel<<<grid, threads, shared_bytes, stream>>>(params);
	XM_CUDA_KERNEL_CHECK();
}
