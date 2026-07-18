"""Optional quantization helpers for inference experiments."""

from __future__ import annotations

from copy import deepcopy

import torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_


def make_torchao_int8_dynamic_model(
    model: torch.nn.Module,
    *,
    compile_model: bool = False,
) -> torch.nn.Module:
    """
    Copy a model and apply torchao dynamic int8 quantization to linear layers.

    The original model is left untouched so benchmarks can compare eager and
    quantized variants with the same inputs.
    """

    quantized_model = deepcopy(model)
    quantized_model.eval()
    quantize_(quantized_model, Int8DynamicActivationInt8WeightConfig())
    if compile_model:
        quantized_model = torch.compile(quantized_model)
    return quantized_model
