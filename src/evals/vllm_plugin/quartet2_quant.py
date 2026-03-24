
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter


@register_quantization_config("quartet2")
class QuartetIIConfig(QuantizationConfig):

    def get_name(self) -> str:
        return "quartet2"

    def get_supported_act_dtypes(self) -> list:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100  # Blackwell (SM 10.0)

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "QuartetIIConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return QuartetIILinearMethod(self)
        return None


class QuartetIILinearMethod(LinearMethodBase):

    def __init__(self, config: QuartetIIConfig):
        self.config = config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from quartet2.quant import quant_fp4, NVFP4QuantMode
        from quartet2.linear import abs_max

        weight = layer.weight.data
        device = weight.device
        out_features = weight.shape[0]

        w_remainder = out_features % 128
        if w_remainder != 0:
            w_pad = 128 - w_remainder
            weight = F.pad(weight, (0, 0, 0, w_pad))
        else:
            w_pad = 0

        mode = NVFP4QuantMode.FOUR_SIX
        weight_amax = abs_max(weight)
        wq = quant_fp4(weight, amax=weight_amax, scale_override=1.0, mode=mode)

        layer.weight_fp4 = wq.fp4
        layer.weight_micro_scales = wq.micro_scales
        layer.weight_tensor_scale = wq.tensor_scale
        layer.w_pad = w_pad

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from quartet2.quant import quant_fp4, NVFP4QuantMode
        from quartet2.linear import abs_max, _fp4_mm

        orig_shape = x.shape
        out_features = layer.weight.shape[0]
        flat_x = x.reshape(-1, x.shape[-1])

        num_rows = flat_x.shape[0]
        remainder = num_rows % 128
        if remainder != 0:
            pad_rows = 128 - remainder
            flat_x = F.pad(flat_x, (0, 0, 0, pad_rows))
        else:
            pad_rows = 0

        input_amax = abs_max(flat_x)
        input_fp4 = quant_fp4(
            flat_x, amax=input_amax,
            scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX,
        )

        alpha = input_fp4.tensor_scale * layer.weight_tensor_scale
        output = _fp4_mm(
            input_fp4.fp4, layer.weight_fp4,
            input_fp4.micro_scales, layer.weight_micro_scales,
            alpha,
        )

        if pad_rows > 0:
            output = output[:num_rows]
        if layer.w_pad > 0:
            output = output[:, :out_features]

        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if bias is not None:
            output = output + bias
        return output
