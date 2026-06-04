// Copyright (c) XMetAI.
// SPDX-License-Identifier: Apache-2.0

#include <onnxruntime_cxx_api.h>

OrtStatus *AddSelectiveScanFnOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base);
OrtStatus *AddMamba3SisoFnOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base);
OrtStatus *AddSoftmaxCpuOp(OrtCustomOpDomain *domain, const OrtApiBase *api_base);

extern "C" OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
	if (!api_base) {
		return nullptr;
	}
	const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
	if (!options) {
		return api->CreateStatus(ORT_INVALID_ARGUMENT, "RegisterCustomOps received null session options");
	}
	OrtCustomOpDomain *xmetai_domain = nullptr;
	OrtStatus *status = api->CreateCustomOpDomain("xmetai_plugins", &xmetai_domain);
	if (status != nullptr) {
		return status;
	}
	status = AddSelectiveScanFnOp(xmetai_domain, api_base);
	if (status != nullptr) {
		return status;
	}
	status = AddMamba3SisoFnOp(xmetai_domain, api_base);
	if (status != nullptr) {
		return status;
	}
	status = AddSoftmaxCpuOp(xmetai_domain, api_base);
	if (status != nullptr) {
		return status;
	}
	return api->AddCustomOpDomain(options, xmetai_domain);
}
