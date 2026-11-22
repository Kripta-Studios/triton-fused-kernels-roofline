import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="Triton kernels need a CUDA GPU")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
