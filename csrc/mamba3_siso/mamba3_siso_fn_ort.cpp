// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include "mamba3_siso.h"
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

class Mamba3SisoKernel {
public:
	Mamba3SisoKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(&api) {
		chunk_size_ = GetIntAttribute(info, "chunk_size", 64);
		has_d_attr_ = GetBoolAttribute(info, "has_d", true);
		has_z_attr_ = GetBoolAttribute(info, "has_z", true);
		if (chunk_size_ <= 0) {
			throw std::runtime_error("Mamba3SisoFn requires chunk_size > 0");
		}
		if (GetBoolAttribute(info, "return_final_states", false)) {
			throw std::runtime_error("Mamba3SisoFn v1 does not support return_final_states");
		}
		if (GetBoolAttribute(info, "has_input_states", false)) {
			throw std::runtime_error("Mamba3SisoFn v1 does not support input states");
		}
		if (GetBoolAttribute(info, "has_cu_seqlens", false)) {
			throw std::runtime_error("Mamba3SisoFn v1 does not support cu_seqlens");
		}
	}

	void Compute(OrtKernelContext *context) {
		TensorView Q = GetTensor(context, 0);
		TensorView K = GetTensor(context, 1);
		TensorView V = GetTensor(context, 2);
		TensorView ADT = GetTensor(context, 3);
		TensorView DT = GetTensor(context, 4);
		TensorView Trap = GetTensor(context, 5);
		TensorView Q_bias = GetTensor(context, 6);
		TensorView K_bias = GetTensor(context, 7);
		TensorView Angles = GetTensor(context, 8);
		TensorView D = GetTensor(context, 9);
		TensorView Z = GetTensor(context, 10);

		ValidateInputs(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z);

		std::vector<int64_t> output_shape = V.shape;
		OrtValue *out_value = GetOutput(context, 0, output_shape);
		float *out_ptr = GetMutableTensorData(out_value);

		const bool has_d = D.elements > 0;
		const bool has_z = Z.elements > 0;

		Mamba3SisoParams params{};
		params.batch = ToIntDim(Q.shape[0], "batch");
		params.seqlen = ToIntDim(Q.shape[1], "seqlen");
		params.nheads_qk = ToIntDim(Q.shape[2], "nheads_qk");
		params.headdim_qk = ToIntDim(Q.shape[3], "headdim_qk");
		params.nheads = ToIntDim(V.shape[2], "nheads");
		params.headdim_v = ToIntDim(V.shape[3], "headdim_v");
		params.headdim_angles = ToIntDim(Angles.shape[3], "headdim_angles");
		params.chunk_size = chunk_size_;
		params.has_d = has_d;
		params.has_z = has_z;
		params.q_ptr = Q.data;
		params.k_ptr = K.data;
		params.v_ptr = V.data;
		params.adt_ptr = ADT.data;
		params.dt_ptr = DT.data;
		params.trap_ptr = Trap.data;
		params.q_bias_ptr = Q_bias.data;
		params.k_bias_ptr = K_bias.data;
		params.angles_ptr = Angles.data;
		params.d_ptr = has_d ? D.data : nullptr;
		params.z_ptr = has_z ? Z.data : nullptr;
		params.out_ptr = out_ptr;

		void *ort_stream = nullptr;
		ThrowIfFailed(api_->KernelContext_GetGPUComputeStream(context, &ort_stream));
		cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_stream);
		mamba3_siso_fwd_cuda(params, stream);
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

	void ValidateInputs(
		const TensorView &Q,
		const TensorView &K,
		const TensorView &V,
		const TensorView &ADT,
		const TensorView &DT,
		const TensorView &Trap,
		const TensorView &Q_bias,
		const TensorView &K_bias,
		const TensorView &Angles,
		const TensorView &D,
		const TensorView &Z) const {
		RequireRank(Q, 4, "Q");
		RequireRank(K, 4, "K");
		RequireRank(V, 4, "V");
		RequireRank(ADT, 3, "ADT");
		RequireRank(DT, 3, "DT");
		RequireRank(Trap, 3, "Trap");
		RequireRank(Q_bias, 2, "Q_bias");
		RequireRank(K_bias, 2, "K_bias");
		RequireRank(Angles, 4, "Angles");
		RequireSameShape(Q, K, "Q", "K");
		RequireSameShape(ADT, DT, "ADT", "DT");
		RequireSameShape(ADT, Trap, "ADT", "Trap");

		const int64_t batch = Q.shape[0];
		const int64_t seqlen = Q.shape[1];
		const int64_t nheads_qk = Q.shape[2];
		const int64_t headdim_qk = Q.shape[3];
		const int64_t nheads = V.shape[2];
		const int64_t headdim_v = V.shape[3];
		const int64_t headdim_angles = Angles.shape[3];

		if (V.shape[0] != batch || V.shape[1] != seqlen) {
			throw std::runtime_error("V batch/seqlen must match Q");
		}
		if (ADT.shape[0] != batch || ADT.shape[1] != nheads || ADT.shape[2] != seqlen) {
			throw std::runtime_error("ADT must have shape (batch, nheads, seqlen)");
		}
		if (Q_bias.shape[0] != nheads || Q_bias.shape[1] != headdim_qk) {
			throw std::runtime_error("Q_bias must have shape (nheads, headdim_qk)");
		}
		if (K_bias.shape != Q_bias.shape) {
			throw std::runtime_error("K_bias shape must match Q_bias");
		}
		if (Angles.shape[0] != batch || Angles.shape[1] != seqlen || Angles.shape[2] != nheads) {
			throw std::runtime_error("Angles must have shape (batch, seqlen, nheads, headdim_angles)");
		}
		if (nheads_qk <= 0 || nheads <= 0 || nheads % nheads_qk != 0) {
			throw std::runtime_error("Mamba3SisoFn requires nheads divisible by nheads_qk");
		}
		if (headdim_qk <= 0 || headdim_qk % 2 != 0) {
			throw std::runtime_error("Mamba3SisoFn requires positive even headdim_qk");
		}
		if (headdim_v <= 0) {
			throw std::runtime_error("Mamba3SisoFn requires positive headdim_v");
		}
		if (headdim_angles <= 0 || headdim_angles > headdim_qk / 2) {
			throw std::runtime_error("Mamba3SisoFn requires 0 < headdim_angles <= headdim_qk / 2");
		}

		const bool has_d = D.elements > 0;
		const bool has_z = Z.elements > 0;
		if (has_d != has_d_attr_) {
			throw std::runtime_error("Mamba3SisoFn has_d attribute does not match D input");
		}
		if (has_z != has_z_attr_) {
			throw std::runtime_error("Mamba3SisoFn has_z attribute does not match Z input");
		}
		if (has_d) {
			RequireRank(D, 1, "D");
			if (D.shape[0] != nheads) {
				throw std::runtime_error("D must have shape (nheads,)");
			}
		}
		if (has_z) {
			RequireRank(Z, 4, "Z");
			if (Z.shape[0] != batch || Z.shape[1] != seqlen || Z.shape[2] != nheads || Z.shape[3] != headdim_v) {
				throw std::runtime_error("Z must have shape (batch, seqlen, nheads, headdim_v)");
			}
		}
	}

	const OrtApi *api_ = nullptr;
	int chunk_size_ = 64;
	bool has_d_attr_ = true;
	bool has_z_attr_ = true;
};

class Mamba3SisoOp : public Ort::CustomOpBase<Mamba3SisoOp, Mamba3SisoKernel> {
public:
	void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
		return new Mamba3SisoKernel(api, info);
	}
	const char *GetName() const { return "Mamba3SisoFn"; }
	size_t GetInputTypeCount() const { return 11; }
	ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	size_t GetOutputTypeCount() const { return 1; }
	ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	const char *GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
};

Mamba3SisoOp g_mamba3_siso_op;

}  // namespace

OrtStatus *AddMamba3SisoFnOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	return api->CustomOpDomain_Add(domain, &g_mamba3_siso_op);
}

OrtStatus *RegisterMamba3SisoFnOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	OrtCustomOpDomain *domain = nullptr;
	OrtStatus *status = api->CreateCustomOpDomain("xmetai_plugins", &domain);
	if (status != nullptr) {
		return status;
	}
	status = AddMamba3SisoFnOp(domain, api_base);
	if (status != nullptr) {
		return status;
	}
	return api->AddCustomOpDomain(options, domain);
}
