// Developed by Wang Zejun

#include "turbo_transformers/layers/bert_mlm_head.h"

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/utils.h"

namespace turbo_transformers {
namespace layers {

void BertMLMHead::operator()(const core::Tensor& input_tensor, core::Tensor* output,
                             const std::string& hidden_act = "gelu") const {
  TT_ENFORCE_EQ(input_tensor.n_dim(), 3, "input's dim should be 3, not %d",
                input_tensor.n_dim());
  TT_ENFORCE_EQ(input_tensor.shape(2), dense_weight_.shape(0),
                "input and weight shape mismatch %d, %d",
                input_tensor.shape(2), dense_weight_.shape(0));

  core::Tensor dense_output(nullptr);
  dense_output.Reshape<float>({ 
      input_tensor.shape(0), input_tensor.shape(1), dense_weight_.shape(1)},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, &dense_output, 0.0);
  
  if (hidden_act == "gelu" || hidden_act == "Gelu" || hidden_act == "GELU") {
    kernels::AddBiasAct<float, kernels::ActivationType::Gelu>(dense_bias_, &dense_output);
  } else if (hidden_act == "relu" || hidden_act == "Relu" || hidden_act == "RELU") {
    kernels::AddBiasAct<float, kernels::ActivationType::Relu>(dense_bias_, &dense_output);
  } else if (hidden_act == "tanh" || hidden_act == "Tanh" || hidden_act == "TANH") {
    kernels::AddBiasAct<float, kernels::ActivationType::Tanh>(dense_bias_, &dense_output);
  } else {
    TT_THROW("hidden_act must be relu, gelu or tanh");
  }

  TT_ENFORCE_EQ(dense_output.shape(2), layer_norm_weight_.shape(0),
                "input and weight shape mismatch %d, %d",
                dense_output.shape(2), layer_norm_weight_.shape(0));
  kernels::LayerNorm<float>(layer_norm_weight_, layer_norm_bias_, &dense_output);

  TT_ENFORCE_EQ(dense_output.shape(2), decoder_weight_.shape(0),
                "input and weight shape mismatch %d, %d",
                dense_output.shape(2), decoder_weight_.shape(0));
  output->Reshape<float>({
      input_tensor.shape(0), input_tensor.shape(1), decoder_weight_.shape(1)},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::MatMul(dense_output, false, decoder_weight_, false, 1.0, output, 0.0);
  kernels::AddBias(decoder_bias_, output);
}

void BertMLMHead::EnforceShapeAndType() const {
  TT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  TT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  TT_ENFORCE_EQ(dense_weight_.shape(1), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d",
                dense_weight_.shape(1), dense_bias_.shape(0));

  TT_ENFORCE_EQ(decoder_weight_.n_dim(), 2, "decoder weight must be matrix");
  TT_ENFORCE_EQ(decoder_bias_.n_dim(), 1, "decoder bias must be vector");
  TT_ENFORCE_EQ(decoder_weight_.shape(1), decoder_bias_.shape(0),
                "weight and bias shape mismatch %d, %d",
                decoder_weight_.shape(1), decoder_bias_.shape(0));

  TT_ENFORCE_EQ(layer_norm_weight_.n_dim(), 1, "layer norm weight must be vector");
  TT_ENFORCE_EQ(layer_norm_bias_.n_dim(), 1, "layer norm bias must be vector");
  TT_ENFORCE_EQ(layer_norm_weight_.shape(0), layer_norm_bias_.shape(0),
                "weight and bias shape mismatch %d, %d",
                layer_norm_weight_.shape(1), layer_norm_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> dense_weight <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> dense_bias <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    os << "<<<<<<<< decoder_weight <<<<<<<<<<";
    decoder_weight_.Print<float>(os);
    os << "<<<<<<<< decoder_bias <<<<<<<<<<";
    decoder_bias_.Print<float>(os);
    os << "<<<<<<<< layer_norm_weight <<<<<<<<<<";
    layer_norm_weight_.Print<float>(os);
    os << "<<<<<<<< layer_norm_bias <<<<<<<<<<";
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

} // namespace layers
} // namespace turbo_transformers

