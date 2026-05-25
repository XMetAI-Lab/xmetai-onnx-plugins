// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#include <onnxruntime_cxx_api.h>

OrtStatus *RegisterSelectiveScanFnOps(OrtSessionOptions *options, const OrtApiBase *api_base);
OrtStatus *RegisterSoftmaxCpuOps(OrtSessionOptions *options, const OrtApiBase *api_base);

extern "C" OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
	if (!api_base) {
		return nullptr;
	}
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	if (!options) {
		return api->CreateStatus(ORT_INVALID_ARGUMENT, "RegisterCustomOps received null session options");
	}
	OrtStatus *status = RegisterSelectiveScanFnOps(options, api_base);
	if (status != nullptr) {
		return status;
	}
	return RegisterSoftmaxCpuOps(options, api_base);
}
