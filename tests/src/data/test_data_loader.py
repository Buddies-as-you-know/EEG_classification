import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import scipy.io

from src.data.data_loader import TrainTestDasetCreate, denoizeing


def test_denoizeing() -> None:
    data = np.random.rand(100, 32, 64)
    denoized_data = denoizeing(data)

    assert isinstance(denoized_data, np.ndarray)
    assert data.shape == denoized_data.shape
