import pytest

pytest.importorskip("torchvision", reason="模型测试需要 torchvision")

import torch

from core.models import BaseClassifier, SimpleUNet


def test_simple_unet_returns_two_image_sized_features():
    model = SimpleUNet().eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        robust, non_robust = model(inputs)

    assert robust.shape == inputs.shape
    assert non_robust.shape == inputs.shape
    torch.testing.assert_close(non_robust, inputs - robust)


def test_base_classifier_returns_six_logits():
    model = BaseClassifier().eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 32, 32))

    assert outputs.shape == (2, 6)
    assert torch.isfinite(outputs).all()
