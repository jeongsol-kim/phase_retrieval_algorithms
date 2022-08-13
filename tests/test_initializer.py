import pytest
from initializer import get_initializer


def test_get_initializer_not_defined_error():
    with pytest.raises(NameError):
        _ = get_initializer(name='not_defined')

def test_gaussian_initializer_shape():
    initializer_fn = get_initializer(name='gaussian')
    shape = (1, 3, 128, 128)
    init_array = initializer_fn(shape)

    assert init_array.shape == shape

def test_gaussian_initializer_scale():
    initializer_fn = get_initializer(name='gaussian')
    shape = (1, 3, 128, 128)
    assert initializer_fn(shape).min() >= 0.0
