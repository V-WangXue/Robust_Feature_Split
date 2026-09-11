import pytest

from config import Config


def test_default_config_is_valid():
    Config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("train_samples", 0),
        ("test_samples", 0),
        ("base_epochs", 0),
        ("separator_epochs", 0),
        ("robust_epochs", 0),
        ("num_workers", -1),
        ("lr_base", 0),
        ("lr_separator", 0),
        ("lr_robust", 0),
        ("epsilon_fgsm", 0),
        ("num_classes", 5),
    ],
)
def test_invalid_config_is_rejected(monkeypatch, field, value):
    monkeypatch.setattr(Config, field, value)
    with pytest.raises(AssertionError):
        Config.validate()


def test_decay_epoch_must_be_inside_training_range(monkeypatch):
    monkeypatch.setattr(Config, "lr_decay_epoch", Config.robust_epochs)
    with pytest.raises(AssertionError, match="学习率衰减轮次"):
        Config.validate()
