import pytest
import torch


@pytest.fixture(autouse=True)
def single_torch_thread():
    """The evaluation kernels in these tests are small; extra intra-op threads
    only add contention (and can be many times slower on a busy machine)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
