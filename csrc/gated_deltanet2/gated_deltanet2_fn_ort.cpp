// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include "gated_deltanet2.h"
#include "../utils/ort_common.h"

namespace {

using xmetai::ort::MakeTensorView;
using xmetai::ort::TensorView;
using xmetai::ort::ThrowIfFailed;

inline void RequireRank(const TensorView &view, size_t rank, const char *name) {
	if (view.shape.size() != rank) {
		throw std::runtime_error(std::string(name) + " must have rank " + std::to_string(rank));
	}
}

inline void RequireSameShape(const TensorView &lhs, const TensorView &rhs, const char *lhs_name, const char *rhs_name) {
	if (lhs.shape != rhs.shape) {
		throw std::runtime_error(std::string(lhs_name) + " shape must match " + rhs_name);
	}
}

inline int ToIntDim(int64_t value, const char *name) {
	if (value <= 0 || value > static_cast<int64_t>(INT32_MAX)) {
		throw std::runtime_error(std::string(name) + " dimension is out of range");
	}
	return static_cast<int>(value);
}

class GatedDeltaNet2Kernel {
public:
	GatedDeltaNet2Kernel(const OrtApi &api, const OrtKernelInfo *info) : api_(&api) {
		scale_ = GetFloatAttribute(info, "scale", 1.0f);
		use_qk_l2norm_in_kernel_ = GetBoolAttribute(info, "use_qk_l2norm_in_kernel", true);
		if (GetBoolAttribute(info, "output_final_state", false)) {
			throw std::runtime_error("GatedDeltaNet2Fn v1 does not support output_final_state");
		}
		if (GetBoolAttribute(info, "use_gate_in_kernel", false)) {
			throw std::runtime_error("GatedDeltaNet2Fn v1 expects pre-applied gate tensors and does not support use_gate_in_kernel");
		}
		if (GetBoolAttribute(info, "has_cu_seqlens", false)) {
			throw std::runtime_error("GatedDeltaNet2Fn v1 does not support cu_seqlens");
		}
		if (GetBoolAttribute(info, "state_v_first", false)) {
			throw std::runtime_error("GatedDeltaNet2Fn v1 expects state layout (batch, head, key, value)");
		}
		if (scale_ <= 0.0f || !std::isfinite(scale_)) {
			throw std::runtime_error("GatedDeltaNet2Fn requires a positive finite scale");
		}
	}

	void Compute(OrtKernelContext *context) {
		TensorView q = GetTensor(context, 0);
		TensorView k = GetTensor(context, 1);
		TensorView v = GetTensor(context, 2);
		TensorView g = GetTensor(context, 3);
		TensorView beta = GetTensor(context, 4);
		TensorView w = GetTensor(context, 5);
		TensorView initial_state = GetTensor(context, 6);

		ValidateInputs(q, k, v, g, beta, w, initial_state);

		std::vector<int64_t> output_shape = v.shape;
		OrtValue *out_value = GetOutput(context, 0, output_shape);
		float *out_ptr = GetMutableTensorData(out_value);

		const int batch = ToIntDim(q.shape[0], "batch");
		const int seqlen = ToIntDim(q.shape[1], "seqlen");
		const int nheads = ToIntDim(q.shape[2], "nheads");
		const int headdim_k = ToIntDim(q.shape[3], "headdim_k");
		const int headdim_v = ToIntDim(v.shape[3], "headdim_v");
		const size_t state_elements = static_cast<size_t>(batch) * nheads * headdim_k * headdim_v;
		const size_t prepared_elements = headdim_k <= kGatedDeltaNet2RegisterKeyLimit
			? static_cast<size_t>(batch) * std::min(seqlen, kGatedDeltaNet2PrepareTokens) * nheads * headdim_k : 0;
		// Each invocation owns its scratch allocation. Use the input's device
		// memory info and ORT's stream-aware scratch API, not raw cudaMalloc or
		// a shared kernel-member cache. Return storage before releasing allocator.
		Ort::Allocator allocator(nullptr);
		auto free_scratch = [&allocator](void *ptr) { if (ptr != nullptr) allocator.Free(ptr); };
		std::unique_ptr<void, decltype(free_scratch)> state_buffer(nullptr, free_scratch);
		std::unique_ptr<void, decltype(free_scratch)> prepared_buffer(nullptr, free_scratch);
		if (state_elements > 0 || prepared_elements > 0) {
			const OrtValue *input = nullptr;
			const OrtMemoryInfo *memory_info = nullptr;
			ThrowIfFailed(api_->KernelContext_GetInput(context, 0, &input));
			ThrowIfFailed(api_->GetTensorMemoryInfo(input, &memory_info));
			OrtAllocator *raw_allocator = nullptr;
			ThrowIfFailed(api_->KernelContext_GetAllocator(context, memory_info, &raw_allocator));
			allocator = Ort::Allocator(raw_allocator);
			void *scratch = nullptr;
			ThrowIfFailed(api_->KernelContext_GetScratchBuffer(context, memory_info, state_elements * sizeof(float), &scratch));
			state_buffer.reset(scratch);
			if (prepared_elements > 0) {
				ThrowIfFailed(api_->KernelContext_GetScratchBuffer(context, memory_info, 4 * prepared_elements * sizeof(float), &scratch));
				prepared_buffer.reset(scratch);
			}
		}

		GatedDeltaNet2Params params{};
		params.batch = batch;
		params.seqlen = seqlen;
		params.nheads = nheads;
		params.headdim_k = headdim_k;
		params.headdim_v = headdim_v;
		params.has_initial_state = initial_state.elements > 0;
		params.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel_;
		params.scale = scale_;
		params.q_ptr = q.data;
		params.k_ptr = k.data;
		params.v_ptr = v.data;
		params.g_ptr = g.data;
		params.b_ptr = beta.data;
		params.w_ptr = w.data;
		params.initial_state_ptr = params.has_initial_state ? initial_state.data : nullptr;
		params.state_ptr = static_cast<float *>(state_buffer.get());
		params.out_ptr = out_ptr;
		if (prepared_elements > 0) {
			params.prepared_q_ptr = static_cast<float *>(prepared_buffer.get());
			params.prepared_k_ptr = params.prepared_q_ptr + prepared_elements;
			params.prepared_decay_ptr = params.prepared_k_ptr + prepared_elements;
			params.prepared_beta_k_ptr = params.prepared_decay_ptr + prepared_elements;
		}

		void *ort_stream = nullptr;
		ThrowIfFailed(api_->KernelContext_GetGPUComputeStream(context, &ort_stream));
		cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_stream);
		gated_deltanet2_fwd_cuda(params, stream);
	}

private:
	TensorView GetTensor(OrtKernelContext *context, size_t index) const {
		const OrtValue *value = nullptr;
		ThrowIfFailed(api_->KernelContext_GetInput(context, index, &value));
		return MakeTensorView(api_, value);
	}

	OrtValue *GetOutput(OrtKernelContext *context, size_t index, const std::vector<int64_t> &shape) const {
		OrtValue *value = nullptr;
		ThrowIfFailed(api_->KernelContext_GetOutput(context, index, shape.data(), shape.size(), &value));
		return value;
	}

	float *GetMutableTensorData(OrtValue *value) const {
		float *ptr = nullptr;
		ThrowIfFailed(api_->GetTensorMutableData(value, reinterpret_cast<void **>(&ptr)));
		return ptr;
	}

	int64_t GetIntAttribute(const OrtKernelInfo *info, const char *name, int64_t default_value) const {
		int64_t value = default_value;
		OrtStatus *status = api_->KernelInfoGetAttribute_int64(info, name, &value);
		if (status != nullptr) {
			api_->ReleaseStatus(status);
			return default_value;
		}
		return value;
	}

	bool GetBoolAttribute(const OrtKernelInfo *info, const char *name, bool default_value) const {
		return GetIntAttribute(info, name, default_value ? 1 : 0) != 0;
	}

	float GetFloatAttribute(const OrtKernelInfo *info, const char *name, float default_value) const {
		float value = default_value;
		OrtStatus *status = api_->KernelInfoGetAttribute_float(info, name, &value);
		if (status != nullptr) {
			api_->ReleaseStatus(status);
			return default_value;
		}
		return value;
	}

	void ValidateInputs(
		const TensorView &q,
		const TensorView &k,
		const TensorView &v,
		const TensorView &g,
		const TensorView &beta,
		const TensorView &w,
		const TensorView &initial_state) const {
		RequireRank(q, 4, "q");
		RequireRank(k, 4, "k");
		RequireRank(v, 4, "v");
		RequireRank(g, 4, "g");
		RequireRank(beta, 4, "beta");
		RequireRank(w, 4, "w");
		RequireSameShape(q, k, "q", "k");
		RequireSameShape(q, g, "q", "g");
		RequireSameShape(q, beta, "q", "beta");

		if (v.shape[0] != q.shape[0] || v.shape[1] != q.shape[1] || v.shape[2] != q.shape[2]) {
			throw std::runtime_error("v batch/seqlen/head dimensions must match q");
		}
		RequireSameShape(v, w, "v", "w");
		if (initial_state.elements > 0) {
			RequireRank(initial_state, 4, "initial_state");
			if (
				initial_state.shape[0] != q.shape[0] ||
				initial_state.shape[1] != q.shape[2] ||
				initial_state.shape[2] != q.shape[3] ||
				initial_state.shape[3] != v.shape[3]) {
				throw std::runtime_error("initial_state must have shape (batch, head, headdim_k, headdim_v)");
			}
		}
	}

	const OrtApi *api_ = nullptr;
	float scale_ = 1.0f;
	bool use_qk_l2norm_in_kernel_ = true;
};

class GatedDeltaNet2Op : public Ort::CustomOpBase<GatedDeltaNet2Op, GatedDeltaNet2Kernel> {
public:
	void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
		return new GatedDeltaNet2Kernel(api, info);
	}
	const char *GetName() const { return "GatedDeltaNet2Fn"; }
	size_t GetInputTypeCount() const { return 7; }
	ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	size_t GetOutputTypeCount() const { return 1; }
	ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	const char *GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
};

GatedDeltaNet2Op g_gated_deltanet2_op;

}  // namespace

OrtStatus *AddGatedDeltaNet2FnOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	return api->CustomOpDomain_Add(domain, &g_gated_deltanet2_op);
}

OrtStatus *RegisterGatedDeltaNet2FnOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	OrtCustomOpDomain *domain = nullptr;
	OrtStatus *status = api->CreateCustomOpDomain("xmetai_plugins", &domain);
	if (status != nullptr) {
		return status;
	}
	status = AddGatedDeltaNet2FnOp(domain, api_base);
	if (status != nullptr) {
		return status;
	}
	return api->AddCustomOpDomain(options, domain);
}
