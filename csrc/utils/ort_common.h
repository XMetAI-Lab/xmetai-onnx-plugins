#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

namespace xmetai::ort {

inline void ThrowIfFailed(OrtStatus *status) {
    if (status != nullptr) {
        Ort::ThrowOnError(status);
    }
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) : ptr_(nullptr) {
        if (bytes == 0) {
            return;
        }
        auto status = cudaMalloc(&ptr_, bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for DeviceBuffer");
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            Release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~DeviceBuffer() { Release(); }

    void *data() const { return ptr_; }

private:
    void Release() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    void *ptr_ = nullptr;
};

struct TensorView {
    const float *data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    size_t elements = 0;
};

inline std::vector<int64_t> MakeRowMajorStrides(const std::vector<int64_t> &shape) {
    std::vector<int64_t> stride(shape.size(), 1);
    int64_t running = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        stride[i] = running;
        running *= shape[i] > 0 ? shape[i] : 1;
    }
    return stride;
}

inline TensorView MakeTensorView(const OrtApi *api, const OrtValue *value) {
    TensorView view;
    OrtTensorTypeAndShapeInfo *info = nullptr;
    ThrowIfFailed(api->GetTensorTypeAndShape(value, &info));
    size_t rank = 0;
    ThrowIfFailed(api->GetDimensionsCount(info, &rank));
    view.shape.resize(rank);
    if (rank > 0) {
        ThrowIfFailed(api->GetDimensions(info, view.shape.data(), rank));
    }
    size_t element_count = 0;
    ThrowIfFailed(api->GetTensorShapeElementCount(info, &element_count));
    view.elements = element_count;
    ONNXTensorElementDataType type;
    ThrowIfFailed(api->GetTensorElementType(info, &type));
    api->ReleaseTensorTypeAndShapeInfo(info);
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("TensorView expects float32 tensors");
    }
    if (view.elements == 0) {
        return view;
    }
    void *ptr = nullptr;
    ThrowIfFailed(api->GetTensorMutableData(const_cast<OrtValue *>(value), &ptr));
    view.data = static_cast<const float *>(ptr);
    view.stride = MakeRowMajorStrides(view.shape);
    return view;
}

}  // namespace xmetai::ort
