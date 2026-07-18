import builtins

import pytest
import torch

from simple_transformer.quantization import make_torchao_int8_dynamic_model


def test_torchao_quantization_helper_reports_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torchao"):
            raise ImportError("missing torchao")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install torchao"):
        make_torchao_int8_dynamic_model(torch.nn.Linear(2, 2))
