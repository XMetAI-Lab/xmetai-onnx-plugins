// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

// Bound the per-thread register arrays. Larger key dimensions retain the
// generic implementation and its global state buffer.
constexpr int kGatedDeltaNet2RegisterKeyLimit = 256;

struct GatedDeltaNet2Params {
	int batch;
	int seqlen;
	int nheads;
	int headdim_k;
	int headdim_v;
	bool has_initial_state;
	bool use_qk_l2norm_in_kernel;
	float scale;

	const float *__restrict__ q_ptr;
	const float *__restrict__ k_ptr;
	const float *__restrict__ v_ptr;
	const float *__restrict__ g_ptr;
	const float *__restrict__ b_ptr;
	const float *__restrict__ w_ptr;
	const float *__restrict__ initial_state_ptr;
	float *__restrict__ state_ptr;
	float *__restrict__ out_ptr;
};

void gated_deltanet2_fwd_cuda(GatedDeltaNet2Params &params, cudaStream_t stream);
