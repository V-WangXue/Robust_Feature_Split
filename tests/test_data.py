import pytest

pytest.importorskip("torchvision", reason="数据管道测试需要 torchvision")

from PIL import Image

from core.data import get_transforms


def test_train_and_test_transforms_have_expected_difference():
    train_transform, test_transform = get_transforms()
    image = Image.new("RGB", (64, 48), color=(128, 64, 32))

    train_tensor = train_transform(image)
    test_tensor = test_transform(image)

    assert train_tensor.shape == (3, 32, 32)
    assert test_tensor.shape == (3, 32, 32)
    assert any(type(op).__name__.startswith("Random") for op in train_transform.transforms)
    assert not any(type(op).__name__.startswith("Random") for op in test_transform.transforms)
