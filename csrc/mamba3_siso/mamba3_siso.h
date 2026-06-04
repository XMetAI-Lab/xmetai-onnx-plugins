// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

struct Mamba3SisoParams {
	using index_t = uint32_t;

	int batch;
	int seqlen;
	int nheads;
	int nheads_qk;
	int headdim_qk;
	int headdim_v;
	int headdim_angles;
	int chunk_size;
	bool has_d;
	bool has_z;

	const float *__restrict__ q_ptr;
	const float *__restrict__ k_ptr;
	const float *__restrict__ v_ptr;
	const float *__restrict__ adt_ptr;
	const float *__restrict__ dt_ptr;
	const float *__restrict__ trap_ptr;
	const float *__restrict__ q_bias_ptr;
	const float *__restrict__ k_bias_ptr;
	const float *__restrict__ angles_ptr;
	const float *__restrict__ d_ptr;
	const float *__restrict__ z_ptr;
	float *__restrict__ out_ptr;
};

void mamba3_siso_fwd_cuda(Mamba3SisoParams &params, cudaStream_t stream);
