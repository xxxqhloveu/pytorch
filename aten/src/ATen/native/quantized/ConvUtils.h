#pragma once
#include <ATen/core/List.h>
#include <ATen/native/ConvUtils.h>

namespace at::native::quantized {

template <int kSpatialDim>
TORCH_API at::SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, kSpatialDim>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation);

} // namespace at::native::quantized
