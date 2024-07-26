// Developed by Wang Zejun

#pragma once
#include <memory>
#include <utility>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class BertMLMHead {
  public:
  BertMLMHead(core::Tensor dense_weight, core::Tensor dense_bias,
              core::Tensor decoder_weight, core::Tensor decoder_bias,
              core::Tensor layer_norm_weight, core::Tensor layer_norm_bias)
    : dense_weight_(std::move(dense_weight)),
      dense_bias_(std::move(dense_bias)),
      decoder_weight_(std::move(decoder_weight)),
      decoder_bias_(std::move(decoder_bias)),
      layer_norm_weight_(std::move(layer_norm_weight)),
      layer_norm_bias_(std::move(layer_norm_bias)) {
    EnforceShapeAndType();
  }

  void EnforceShapeAndType() const;

  void operator()(const core::Tensor& input_tensor, core::Tensor* output,
                  const std::string& hidden_act = "gelu") const;

  private:
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
  core::Tensor decoder_weight_;
  core::Tensor decoder_bias_;
  core::Tensor layer_norm_weight_;
  core::Tensor layer_norm_bias_;
};

} // namespace layers
} // namespace turbo_transformers

