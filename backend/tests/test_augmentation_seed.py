import sys
import types


class _StubTransform:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubToTensorV2:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _factory(*args, **kwargs):
    return _StubTransform(*args, **kwargs)


albumentations = types.ModuleType("albumentations")
for name in [
    "Affine",
    "CLAHE",
    "Compose",
    "ElasticTransform",
    "GaussNoise",
    "GaussianBlur",
    "HorizontalFlip",
    "HueSaturationValue",
    "ISONoise",
    "LongestMaxSize",
    "MotionBlur",
    "Normalize",
    "PadIfNeeded",
    "Perspective",
    "RandomBrightnessContrast",
    "OneOf",
]:
    setattr(albumentations, name, _factory)

albumentations_pytorch = types.ModuleType("albumentations.pytorch")
albumentations_pytorch.ToTensorV2 = _StubToTensorV2

sys.modules.setdefault("albumentations", albumentations)
sys.modules.setdefault("albumentations.pytorch", albumentations_pytorch)

from backend.codex_pipeline.data import augmentation


def test_train_transforms_forward_seed_to_albumentations_compose(monkeypatch):
    calls = []

    def fake_compose(transforms, **kwargs):
        calls.append(kwargs)
        return {"transforms": transforms, "kwargs": kwargs}

    monkeypatch.setattr(augmentation.A, "Compose", fake_compose)

    augmentation.get_train_transform({}, seed=17)
    augmentation.get_train_transform_numpy({}, seed=23)

    assert calls == [{"seed": 17}, {"seed": 23}]
