from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, default_loader
import os
import cv2
import torch

'''
Use OpenCV as backend and support use half images of the dataset
'''

def make_dataset_custom(directory, class_to_idx, extensions=None, is_valid_file=None, half_data=False):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        temp = False
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    if half_data is False or temp:
                        instances.append(item)
                    temp = not temp
    return instances


class DatasetFolderCustom(DatasetFolder):
    """ Half of the dataset
    """
    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, num_cls=1000, half_data=False):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        if len(classes) != 1000:
            assert f"Found {len(classes)} classes"
        if num_cls != 1000:
            classes = classes[:num_cls]
            class_to_idx = {k: class_to_idx[k] for k in classes if k in class_to_idx}
        samples = make_dataset_custom(self.root, class_to_idx, extensions, is_valid_file, half_data)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.loader == "opencv":
            path, target = self.samples[index]
            sample = cv2.imread(path)
            
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if len(sample) > 1 and sample[1] is None:
                return sample[0], target # For DCTNet

            return sample, target
        elif self.loader == "wpt":
            path, target = self.samples[index]
            sample = torch.load(path)

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        else:
            return super(DatasetFolderCustom, self).__getitem__(index)


class ImageFolderCustom(DatasetFolderCustom):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, num_cls=1000, half_data=False):
        
        extensions = IMG_EXTENSIONS + ('.pt',)
        super(ImageFolderCustom, self).__init__(root, loader, extensions if is_valid_file is None else None,
                                                transform=transform,
                                                target_transform=target_transform,
                                                is_valid_file=is_valid_file,
                                                num_cls=num_cls,
                                                half_data=half_data)
        self.imgs = self.samples
