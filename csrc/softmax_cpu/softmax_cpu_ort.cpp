#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "../utils/ort_common.h"

namespace {

using xmetai::ort::MakeTensorView;
using xmetai::ort::TensorView;
using xmetai::ort::ThrowIfFailed;

class SoftmaxCpuKernel {
public:
	SoftmaxCpuKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(&api) {
		axis_ = GetAxisAttribute(info);
	}

	void Compute(OrtKernelContext *context) {
		const OrtValue *input = nullptr;
		ThrowIfFailed(api_->KernelContext_GetInput(context, 0, &input));
		TensorView view = MakeTensorView(api_, input);
		std::vector<int64_t> output_shape = view.shape;
		OrtValue *output = nullptr;
		ThrowIfFailed(api_->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output));
		if (view.elements == 0) {
			return;
		}
		float *out = GetMutableTensorData(output);
		const float *in = view.data;

		const int64_t rank = static_cast<int64_t>(view.shape.size());
		int64_t axis = axis_;
		if (axis < 0) {
			axis += rank;
		}
		if (axis < 0 || axis >= rank) {
			throw std::runtime_error("SoftmaxCPU axis out of range");
		}

		size_t outer = 1;
		for (int64_t i = 0; i < axis; ++i) {
			outer *= static_cast<size_t>(view.shape[i]);
		}
		size_t axis_dim = static_cast<size_t>(view.shape[axis]);
		size_t inner = 1;
		for (int64_t i = axis + 1; i < rank; ++i) {
			inner *= static_cast<size_t>(view.shape[i]);
		}

		const size_t stride = inner;
		const size_t matrix_stride = axis_dim * inner;
		for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
			for (size_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
				const size_t base = outer_idx * matrix_stride + inner_idx;
				float max_val = -std::numeric_limits<float>::infinity();
				for (size_t a = 0; a < axis_dim; ++a) {
					const size_t offset = base + a * stride;
					max_val = std::max(max_val, in[offset]);
				}
				float sum = 0.0f;
				for (size_t a = 0; a < axis_dim; ++a) {
					const size_t offset = base + a * stride;
					float exp_val = std::exp(in[offset] - max_val);
					out[offset] = exp_val;
					sum += exp_val;
				}
				const float inv_sum = sum > 0 ? 1.0f / sum : 0.0f;
				for (size_t a = 0; a < axis_dim; ++a) {
					const size_t offset = base + a * stride;
					out[offset] *= inv_sum;
				}
			}
		}
	}

private:
	int64_t GetAxisAttribute(const OrtKernelInfo *info) const {
		int64_t axis = -1;
		OrtStatus *status = api_->KernelInfoGetAttribute_int64(info, "axis", &axis);
		if (status != nullptr) {
			api_->ReleaseStatus(status);
			axis = -1;
		}
		return axis;
	}

	float *GetMutableTensorData(OrtValue *value) const {
		float *ptr = nullptr;
		ThrowIfFailed(api_->GetTensorMutableData(value, reinterpret_cast<void **>(&ptr)));
		return ptr;
	}

	const OrtApi *api_ = nullptr;
	int64_t axis_ = -1;
};

class SoftmaxCpuOp : public Ort::CustomOpBase<SoftmaxCpuOp, SoftmaxCpuKernel> {
public:
	void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const { return new SoftmaxCpuKernel(api, info); }
	const char *GetName() const { return "SoftmaxCPU"; }
	size_t GetInputTypeCount() const { return 1; }
	ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	size_t GetOutputTypeCount() const { return 1; }
	ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
	const char *GetExecutionProviderType() const { return "CPUExecutionProvider"; }
};

SoftmaxCpuOp g_softmax_cpu_op;

}  // namespace

OrtStatus *AddSoftmaxCpuOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	return api->CustomOpDomain_Add(domain, &g_softmax_cpu_op);
}

OrtStatus *RegisterSoftmaxCpuOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	OrtCustomOpDomain *domain = nullptr;
	OrtStatus *status = api->CreateCustomOpDomain("xmetai_plugins", &domain);
	if (status != nullptr) {
		return status;
	}
	status = AddSoftmaxCpuOp(domain, api_base);
	if (status != nullptr) {
		return status;
	}
	return api->AddCustomOpDomain(options, domain);
}
