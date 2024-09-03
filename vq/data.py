import glob
import os
from typing import Callable

import numpy as np
from datasets import Dataset, Image

from torchutils import local_seed_numpy

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def is_image_file(path: str) -> bool:
    """Return whether if the path is a PIL.Image.Image openable file.

    Args:
    ----
        path (str): the path to an image.

    Returns:
    -------
        bool: file?

    """
    ext = {os.path.splitext(os.path.basename(path))[-1].lower()}
    return ext.issubset(IMG_EXTENSIONS)


def imagepaths(
    paths: list[str],
    transforms: Callable,
    num_images: int | None = None,
    filter_fn: Callable | None = None,
    seed: int | None = None,
):
    """Create `dataset.Dataset` object.

    The arguments are made equivalent to `dataset.ImageFolder` class.

    Args:
    ----
        paths (list[str]): list of paths to image files.
        transforms (Callable): A callable that transforms the image.
        num_images (int | None, optional): If given, the dataset will be reduced to have at most num_images samples.
            Default: None.
        filter_fn (Callable | None, optional): A callable that inputs a path and returns a bool to filter the files.
            Default: None.
        seed (int | None): seed for random sampling when reducing data according to `num_images`. Default: None.

    Returns:
    -------
        Dataset: The created dataset.

    """

    def generator():
        for path in paths:
            if is_image_file(path):
                yield dict(image=path)

    dataset = Dataset.from_generator(generator)
    dataset = dataset.sort('image')  # always sort the data.

    if callable(filter_fn):
        dataset = dataset.filter(filter_fn)

    total_images = len(dataset)
    if num_images is not None and num_images < total_images:
        # Reduce dataset size using random permutation.
        with local_seed_numpy(seed=seed, enabled=seed is not None):
            permutation = np.random.permutation(total_images)[:num_images]
        # Sort indices to keep the dataset order.
        permutation = np.sort(permutation)
        dataset = dataset.select(permutation)

    dataset = dataset.cast_column('image', Image(mode='RGB'))

    def transform_sample(samples):
        samples['image'] = [transforms(image) for image in samples['image']]
        return samples

    dataset = dataset.with_transform(transform_sample)

    return dataset


def imagefolder(
    data_root: str,
    transforms: Callable,
    num_images: int | None = None,
    filter_fn: Callable | None = None,
    seed: int | None = None,
):
    """Create `dataset.Dataset` object.

    The arguments are made equivalent to `dataset.ImageFolder` class.

    Args:
    ----
        data_root (str): Root directory of images. Images are searched recursively inside this folder.
        transforms (Callable): A callable that transforms the image.
        num_images (int | None, optional): If given, the dataset will be reduced to have at most num_images samples.
            Default: None.
        filter_fn (Callable | None, optional): A callable that inputs a path and returns a bool to filter the files.
            Default: None.
        seed (int | None): seed for random sampling when reducing data according to `num_images`. Default: None.

    Returns:
    -------
        Dataset: The created dataset.

    """
    return imagepaths(
        glob.glob(os.path.join(data_root, '**', '*'), recursive=True),
        transforms=transforms,
        num_images=num_images,
        filter_fn=filter_fn,
        seed=seed,
    )
