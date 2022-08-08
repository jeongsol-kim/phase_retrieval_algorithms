from dataloader import get_valid_png_loader

def test_loader_data_shape():
    root = '/data/FFHQ/'
    batch_size = 2

    loader = get_valid_png_loader(root, batch_size)
    image = next(iter(loader))
    assert image.shape == (batch_size, 3, 256, 256)

def test_loader_data_scale():
    root = '/data/FFHQ/'
    batch_size = 1

    loader = get_valid_png_loader(root, batch_size)
    image = next(iter(loader))
    assert image.min() >= 0.0 and image.max() <= 1.0
