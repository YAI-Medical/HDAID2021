from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from torch import Tensor
    PathType = Union[str, os.PathLike]
    T = TypeVar('T')


def pil_gray_loader(path: "PathType") -> "Image.Image":
    with open(path, 'rb') as fp:
        return Image.open(fp).convert('L')


def numpy_loader(path: "PathType") -> "np.ndarray":
    return np.load(path).astype(float)


class ImageList(Dataset):

    samples: "Sequence[Tuple[PathType, PathType]]"
    transform: "Optional[Callable[[PathType], ...]]"
    target_transform: "Optional[Callable[[PathType], ...]]"
    loader: "Optional[Callable[[PathType], ...]]"
    target_loader: "Optional[Callable[[PathType], ...]]"
    num_classes: "Optional[int]"
    one_hot: "bool"

    def __init__(
            self,
            samples: "Sequence[Tuple[PathType, PathType]]",
            transform: "Optional[Callable[..., ...]] " = None,
            target_transform: "Optional[Callable[..., ...]]" = None,
            loader: "Optional[Callable[[PathType], ...]]" = pil_gray_loader,
            target_loader: "Optional[Callable[[PathType], ...]]" = numpy_loader,
            num_classes: "Optional[int]" = 2,
            one_hot: "bool" = False
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.target_loader = target_loader
        self.num_classes = num_classes
        self.one_hot = one_hot

    @classmethod
    def from_path(
            cls: "Type[T]",
            directory: "PathType",
            image_extension: "str" = '.png',
            target_extension: "str" = '.npy',
            **kwargs: ...,
    ) -> "T":
        return cls(make_dataset(directory, image_extension, target_extension), **kwargs)

    def __getitem__(self, index: "Union[slice, int]") -> "Union[Subset, Tuple[Tensor, Tensor]]":
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
        if self.one_hot:
            target = self.get_one_hot_encoded_target(target)
        return sample, target

    def __len__(self) -> "int":
        return len(self.samples)

    def get_one_hot_encoded_target(self, target_img: "Tensor") -> "Tensor":
        assert isinstance(self.num_classes, int)
        from models.functional import one_hot_nd
        return one_hot_nd(target_img, self.num_classes, nd=2)


def make_dataset(
        directory: "[str, os.PathLike]",
        image_extension: "str" = '.png',
        target_extension: "str" = '.npy',
) -> "List[Tuple[PathType, PathType]]":
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
    directory = os.path.expanduser(str(directory))
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


del TYPE_CHECKING

__all__ = ['ImageList', 'make_dataset']
