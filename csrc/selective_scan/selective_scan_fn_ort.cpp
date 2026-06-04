#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include "selective_scan.h"
#include "../utils/ort_common.h"

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

namespace {

using xmetai::ort::DeviceBuffer;
using xmetai::ort::MakeRowMajorStrides;
using xmetai::ort::MakeTensorView;
using xmetai::ort::TensorView;
using xmetai::ort::ThrowIfFailed;

inline void MaybeInsertGroupDim(TensorView &view) {
	if (view.shape.size() == 3) {
		view.shape.insert(view.shape.begin() + 1, 1);
		view.stride = MakeRowMajorStrides(view.shape);
	}
}

inline std::vector<int64_t> EnsureShapeRank(const TensorView &view, size_t expected_rank, const char *name) {
	if (view.shape.size() != expected_rank) {
		throw std::runtime_error(std::string(name) + " must have rank " + std::to_string(expected_rank));
	}
	return view.shape;
}

inline uint32_t ToUint32(int64_t value, const char *what) {
	if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error(std::string("Value out of range for ") + what);
	}
	return static_cast<uint32_t>(value);
}


class SelectiveScanKernel {
public:
	SelectiveScanKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(&api) {
		delta_softplus_ = GetBoolAttribute(info, "delta_softplus", false);
		return_last_state_ = GetBoolAttribute(info, "return_last_state", false);
		if (return_last_state_) {
			throw std::runtime_error("SelectiveScanFn does not support return_last_state=1 in ORT kernel");
		}
	}

	void Compute(OrtKernelContext *context) {
		const OrtValue *u_input = GetInput(context, 0);
		const OrtValue *delta_input = GetInput(context, 1);
		const OrtValue *a_input = GetInput(context, 2);
		const OrtValue *b_input = GetInput(context, 3);
		const OrtValue *c_input = GetInput(context, 4);
		const OrtValue *d_input = GetInput(context, 5);
		const OrtValue *z_input = GetInput(context, 6);
		const OrtValue *delta_bias_input = GetInput(context, 7);

		TensorView u = MakeTensorView(api_, u_input);
		TensorView delta = MakeTensorView(api_, delta_input);
		TensorView A = MakeTensorView(api_, a_input);
		TensorView B = MakeTensorView(api_, b_input);
		TensorView C = MakeTensorView(api_, c_input);
		TensorView D = MakeTensorView(api_, d_input);
		TensorView z = MakeTensorView(api_, z_input);
		TensorView delta_bias = MakeTensorView(api_, delta_bias_input);

		MaybeInsertGroupDim(B);  // rearrange(B, "b dstate l -> b 1 dstate l") if B.ndim == 3
		MaybeInsertGroupDim(C);  // rearrange(C, "b dstate l -> b 1 dstate l") if C.ndim == 3

		if (u.shape.size() != 3) {
			throw std::runtime_error("u must be 3-D");
		}
		if (delta.shape != u.shape) {
			throw std::runtime_error("delta must have same shape as u");
		}
		if (A.shape.size() != 2) {
			throw std::runtime_error("A must be 2-D");
		}
		if (B.shape.size() < 2) {
			throw std::runtime_error("B must have rank >= 2");
		}
		if (C.shape.size() < 2) {
			throw std::runtime_error("C must have rank >= 2");
		}

		const int batch = static_cast<int>(u.shape[0]);
		const int dim = static_cast<int>(u.shape[1]);
		const int seqlen = static_cast<int>(u.shape[2]);
		const int dstate = static_cast<int>(A.shape[1]);
		if (A.shape[0] != dim) {
			throw std::runtime_error("A dimension mismatch with u");
		}
		if (dstate > 256) {
			throw std::runtime_error("selective_scan supports d_state up to 256");
		}

		const bool is_variable_B = B.shape.size() >= 3;
		const bool is_variable_C = C.shape.size() >= 3;
		const int n_groups = is_variable_B ? static_cast<int>(B.shape[1]) : 1;
		if (dim % n_groups != 0) {
			throw std::runtime_error("dim must be divisible by n_groups");
		}
		const int n_chunks = std::max(1, (seqlen + 2048 - 1) / 2048);

		const bool has_D = D.elements > 0;
		const bool has_z = z.elements > 0;
		const bool has_delta_bias = delta_bias.elements > 0;

		OrtValue *out_value = GetOutput(context, 0, u.shape);
		float *out_ptr = GetMutableTensorData(out_value);
		std::vector<int64_t> out_stride = MakeRowMajorStrides(u.shape);

		DeviceBuffer x_buffer(static_cast<size_t>(batch) * dim * n_chunks * dstate * 2 * sizeof(float));
		DeviceBuffer out_z_buffer(has_z ? z.elements * sizeof(float) : 0);  // allocate only if needed

		SSMParamsBase params{};
		params.batch = batch;
		params.dim = dim;
		params.seqlen = seqlen;
		params.dstate = dstate;
		params.n_groups = n_groups;
		params.n_chunks = n_chunks;
		params.dim_ngroups_ratio = dim / n_groups;
		params.delta_softplus = delta_softplus_;
		params.is_variable_B = is_variable_B;
		params.is_variable_C = is_variable_C;

		params.A_ptr = const_cast<float *>(A.data);
		params.B_ptr = const_cast<float *>(B.data);
		params.C_ptr = const_cast<float *>(C.data);
		params.D_ptr = has_D ? const_cast<float *>(D.data) : nullptr;
		params.u_ptr = const_cast<float *>(u.data);
		params.delta_ptr = const_cast<float *>(delta.data);
		params.delta_bias_ptr = has_delta_bias ? const_cast<float *>(delta_bias.data) : nullptr;
		params.out_ptr = out_ptr;
		params.x_ptr = x_buffer.data();
		params.z_ptr = has_z ? const_cast<float *>(z.data) : nullptr;
		params.out_z_ptr = has_z ? out_z_buffer.data() : nullptr;

		params.A_d_stride = ToUint32(A.stride[0], "A_d_stride");
		params.A_dstate_stride = ToUint32(A.stride[1], "A_dstate_stride");

		if (!is_variable_B) {
			params.B_d_stride = ToUint32(B.stride[0], "B_d_stride");
			params.B_dstate_stride = ToUint32(B.stride[1], "B_dstate_stride");
		} else {
			params.B_batch_stride = ToUint32(B.stride[0], "B_batch_stride");
			params.B_group_stride = ToUint32(B.stride[1], "B_group_stride");
			params.B_dstate_stride = ToUint32(B.stride.size() > 2 ? B.stride[2] : 1, "B_dstate_stride");
		}

		if (!is_variable_C) {
			params.C_d_stride = ToUint32(C.stride[0], "C_d_stride");
			params.C_dstate_stride = ToUint32(C.stride[1], "C_dstate_stride");
		} else {
			params.C_batch_stride = ToUint32(C.stride[0], "C_batch_stride");
			params.C_group_stride = ToUint32(C.stride[1], "C_group_stride");
			params.C_dstate_stride = ToUint32(C.stride.size() > 2 ? C.stride[2] : 1, "C_dstate_stride");
		}

		params.u_batch_stride = ToUint32(u.stride[0], "u_batch_stride");
		params.u_d_stride = ToUint32(u.stride[1], "u_d_stride");
		params.delta_batch_stride = ToUint32(delta.stride[0], "delta_batch_stride");
		params.delta_d_stride = ToUint32(delta.stride[1], "delta_d_stride");

		if (has_z) {
			params.z_batch_stride = ToUint32(z.stride[0], "z_batch_stride");
			params.z_d_stride = ToUint32(z.stride[1], "z_d_stride");
			params.out_z_batch_stride = params.z_batch_stride;
			params.out_z_d_stride = params.z_d_stride;
		}

		params.out_batch_stride = ToUint32(out_stride[0], "out_batch_stride");
		params.out_d_stride = ToUint32(out_stride[1], "out_d_stride");

		void *ort_stream = nullptr;
		ThrowIfFailed(api_->KernelContext_GetGPUComputeStream(context, &ort_stream));
		cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_stream);
        //sync
        cudaStreamSynchronize(stream);
		selective_scan_fwd_cuda<float, float>(params, stream);
        cudaStreamSynchronize(stream);
		auto cuda_err = cudaGetLastError();
		if (cuda_err != cudaSuccess) {
			throw std::runtime_error("SelectiveScanFn kernel launch failed");
		}
	}



private:
	const OrtValue *GetInput(OrtKernelContext *context, size_t index) const {
		const OrtValue *value = nullptr;
		ThrowIfFailed(api_->KernelContext_GetInput(context, index, &value));
		return value;
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

	bool GetBoolAttribute(const OrtKernelInfo *info, const char *name, bool default_value) const {
		int64_t value = 0;
		OrtStatus *status = api_->KernelInfoGetAttribute_int64(info, name, &value);
		if (status != nullptr) {
			api_->ReleaseStatus(status);
			return default_value;
		}
		return value != 0;
	}

	const OrtApi *api_ = nullptr;
	bool delta_softplus_ = false;
	bool return_last_state_ = false;
};

class SelectiveScanOp : public Ort::CustomOpBase<SelectiveScanOp, SelectiveScanKernel> {
public:
	void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
		return new SelectiveScanKernel(api, info);
	}
	const char *GetName() const { return "SelectiveScanFn"; }
	size_t GetInputTypeCount() const { return 8; }
	ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	size_t GetOutputTypeCount() const { return 1; }
	ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	const char *GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
};

SelectiveScanOp g_selective_scan_op;

}  // namespace

OrtStatus *AddSelectiveScanFnOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	return api->CustomOpDomain_Add(domain, &g_selective_scan_op);
}

OrtStatus *RegisterSelectiveScanFnOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	OrtCustomOpDomain *domain = nullptr;
	OrtStatus *status = api->CreateCustomOpDomain("xmetai_plugins", &domain);
	if (status != nullptr) {
		return status;
	}
	status = AddSelectiveScanFnOp(domain, api_base);
	if (status != nullptr) {
		return status;
	}
	return api->AddCustomOpDomain(options, domain);
}
