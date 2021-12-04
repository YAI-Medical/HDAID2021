from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np
import os


def pil_gray_loader(path):
    with open(path, 'rb') as fp:
        return Image.open(fp).convert('L')


numpy_loader = np.load


class ImageList(Dataset):

    def __init__(
            self, samples,
            transform=None, target_transform=None, num_classes=2,
            loader=pil_gray_loader, target_loader=numpy_loader
    ):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.target_loader = target_loader
        self.num_classes = num_classes

    @classmethod
    def from_path(
            cls,
            directory: "[str, os.PathLike]",
            image_extension: str = '.png',
            target_extension: str = '.npy',
            **kwargs: ...,
    ):
        return cls(make_dataset(directory, image_extension, target_extension), **kwargs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        sample, target = self.samples[index]
        sample = self.loader(sample)
        target = self.target_loader(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.get_one_hot_encoded_target(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def get_one_hot_encoded_target(self, target_img):
        target = target_img.squeeze()
        one_hot_target = np.zeros((self.num_classes, target.size(0), target.size(1)))

        # the target not correctly normalize.
        one_hot_target[0, :, :] = np.where(target == 0, 1, 0)
        one_hot_target[1, :, :] = np.where(target > 0, 1, 0)

        return one_hot_target


def make_dataset(
        directory: "[str, os.PathLike]",
        image_extension: str = '.png',
        target_extension: str = '.npy',
):
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        image_extension (optional): Allowed extension of image.
        target_extension (optional): Allowed extension of target.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    def remove_ext(s):
        return os.path.splitext(s)[0]
    assert all([
        isinstance(directory, (str, os.PathLike)),
        isinstance(image_extension, str),
        isinstance(target_extension, str),
        image_extension.startswith('.'),
        target_extension.startswith('.'),
    ])
    images = set()
    targets = set()
    directory = os.path.expanduser(directory)
    filenames = sorted(os.listdir(directory))
    for filename in sorted(filenames):
        if filename.lower().endswith(image_extension):
            images.add(filename)
        elif filename.lower().endswith(target_extension):
            targets.add(filename)
    common = sorted(set(map(remove_ext, images)).intersection(set(map(remove_ext, targets))))
    return [
        (os.path.join(directory, fn + image_extension), os.path.join(directory, fn + target_extension))
        for fn in common
    ]


__all__ = ['ImageList', 'make_dataset']
