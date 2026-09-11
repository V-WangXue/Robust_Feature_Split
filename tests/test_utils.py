import torch
import torch.nn as nn

from utils import create_dirs, load_model, save_model


def test_save_and_load_model_round_trip(tmp_path):
    source = nn.Linear(4, 2)
    target = nn.Linear(4, 2)
    path = tmp_path / "nested" / "model.pth"

    save_model(source, str(path))
    load_model(target, str(path))

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_param, target_param)


def test_create_dirs_is_idempotent(tmp_path):
    path = tmp_path / "a" / "b"
    create_dirs([str(path)])
    create_dirs([str(path)])
    assert path.is_dir()
