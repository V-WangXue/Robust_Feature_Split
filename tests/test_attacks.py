import pytest

pytest.importorskip("torchvision", reason="攻击测试需要加载项目模型模块")

import torch
import torch.nn as nn

from core.train_robust import _normalized_bounds, fgsm_attack, pgd_attack


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

    def forward(self, x):
        return self.layers(x)


def _standardized_inputs(batch_size=4):
    raw = torch.rand(batch_size, 3, 8, 8)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (raw - mean) / std


def test_normalized_bounds_match_image_net_coordinates():
    inputs = _standardized_inputs()
    lower, upper, std = _normalized_bounds(inputs)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

    torch.testing.assert_close(lower, -mean / std)
    torch.testing.assert_close(upper, (1 - mean) / std)


def test_fgsm_uses_pixel_epsilon_per_channel_and_preserves_valid_range():
    model = TinyClassifier()
    inputs = _standardized_inputs()
    labels = torch.tensor([0, 1, 2, 3])
    epsilon = 8 / 255

    adversarial = fgsm_attack(model, inputs, labels, epsilon)
    lower, upper, std = _normalized_bounds(inputs)

    assert adversarial.shape == inputs.shape
    assert torch.all(adversarial >= lower - 1e-6)
    assert torch.all(adversarial <= upper + 1e-6)
    assert torch.all((adversarial - inputs).abs() <= epsilon / std + 1e-6)


def test_pgd_stays_inside_epsilon_ball_and_is_detached():
    model = TinyClassifier()
    inputs = _standardized_inputs()
    labels = torch.tensor([0, 1, 2, 3])
    epsilon = 8 / 255

    adversarial = pgd_attack(model, inputs, labels, eps=epsilon, alpha=2 / 255, iters=3)
    lower, upper, std = _normalized_bounds(inputs)

    assert not adversarial.requires_grad
    assert torch.all(adversarial >= lower - 1e-6)
    assert torch.all(adversarial <= upper + 1e-6)
    assert torch.all((adversarial - inputs).abs() <= epsilon / std + 1e-5)
