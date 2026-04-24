from src.device import get_device


def test_get_device_returns_string():
    device = get_device()
    assert device in ("cuda", "cpu")


def test_get_device_matches_torch_availability():
    import torch
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device() == expected
