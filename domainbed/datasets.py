# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import os
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
] + [
    'ColoredMNIST_IRM',
    'CelebA_Blond',
    'NICO_Mixed',
    'ImageNet_A',
    'ImageNet_R',
    'ImageNet_V2',
    'ImageNet_V2_Super400',
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        get_normalize(),
    ])


def get_augment_transform(scheme_name='default', input_size=224):
    schemes = {}
    schemes['default'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        get_normalize(),
    ])
    schemes['jigen'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        get_normalize(),
    ])
    if scheme_name not in schemes:
        raise KeyError(f'no such data augmentation scheme: {scheme_name}')
    return schemes[scheme_name]


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class ColoredMNIST_IRM(MultipleDomainDataset):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    def __init__(self, root, test_envs, hparams):
        if 'data_augmentation_scheme' in hparams:
            raise NotImplementedError
        super().__init__()
        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        for i, env in enumerate([0.1, 0.2]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(images, labels, env))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_dataset(images, labels, 0.9))

        self.input_shape = (2, 14, 14)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        if 'data_augmentation_scheme' in hparams:
            raise NotImplementedError
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        if 'data_augmentation_scheme' in hparams:
            raise NotImplementedError
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = get_transform()
        augment_scheme = hparams.get('data_augmentation_scheme', 'default')
        augment_transform = get_augment_transform(augment_scheme)

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = get_transform()
        augment_scheme = hparams.get('data_augmentation_scheme', 'default')
        augment_transform = get_augment_transform(augment_scheme)

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)
        for dataset in self.datasets:
            if not hasattr(dataset, 'samples'):
                image_files = [dataset.dataset._input_array[i] for i in dataset.indices]
                labels = dataset.dataset.y_array[dataset.indices].tolist()
                setattr(dataset, 'samples', list(zip(image_files, labels)))
            else:
                raise Exception


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)
        for dataset in self.datasets:
            if not hasattr(dataset, 'samples'):
                image_files = [dataset.dataset.full_idxs[i] for i in dataset.indices]
                labels = dataset.dataset.y_array[dataset.indices].tolist()
                setattr(dataset, 'samples', list(zip(image_files, labels)))
            else:
                raise Exception


class CelebA_Environment(Dataset):
    def __init__(self, target_attribute_id, split_csv, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        file_names = []
        attributes = []
        with open(split_csv) as f:
            reader = csv.reader(f)
            next(reader)  # discard header
            for row in reader:
                file_names.append(row[0])
                attributes.append(np.array(row[1:], dtype=int))
        attributes = np.stack(attributes, axis=0)
        self.samples = list(zip(file_names, list(attributes[:, target_attribute_id])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, label = self.samples[index]
        image = Image.open(Path(self.img_dir, file_name))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


class CelebA_Blond(MultipleDomainDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['tr_env1', 'tr_env2', 'te_env']
    TARGET_ATTRIBUTE_ID = 9
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if 'data_augmentation_scheme' in hparams:
            raise NotImplementedError(
                'CelebA_Blond has its own data augmentation scheme')

        transform = transforms.Compose([
            transforms.CenterCrop(178),  # crop the face at the center, no stretching
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            get_normalize(),
        ])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0),
                                         ratio=(1.0, 1.3333333333333333)),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),  # do not alter hue
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_normalize(),
        ])

        img_dir = Path(root, 'celeba', 'img_align_celeba')
        self.datasets = []
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            split_csv = Path(root, 'celeba', 'blond_split', f'{env_name}.csv')
            dataset = CelebA_Environment(self.TARGET_ATTRIBUTE_ID, split_csv, img_dir,
                                         env_transform)
            self.datasets.append(dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2  # blond or not


class NICO_Mixed_Environment(Dataset):
    def __init__(self, split_csv, img_root_dir, transform=None):
        self.transform = transform
        self.samples = []
        with open(split_csv) as f:
            reader = csv.reader(f)
            for img_path, category_name, context_name, superclass in reader:
                img_path = img_path.replace('\\', '/')
                img_path = Path(img_root_dir, superclass, 'images', img_path)
                self.samples.append((img_path, {'animal': 0, 'vehicle': 1}[superclass]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


class NICO_Mixed(MultipleDomainDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['train1', 'train2', 'val', 'test']
    def __init__(self, root, test_envs, hparams):
        super().__init__()

        transform = get_transform()
        augment_scheme = hparams.get('data_augmentation_scheme', 'default')
        augment_transform = get_augment_transform(augment_scheme)

        self.datasets = []
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            split_csv = Path(root, 'NICO', 'mixed_split_corrected',
                             f'env_{env_name}.csv')
            dataset = NICO_Mixed_Environment(split_csv, Path(root, 'NICO'),
                                             env_transform)
            self.datasets.append(dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2  # animal or vehicle


class CustomLabelMappingDataset(ImageFolder):
    def __init__(self, root, classes, subclass_to_idx, transform=None):
        self.classes = classes
        self.subclass_to_idx = subclass_to_idx
        super(CustomLabelMappingDataset, self).__init__(root, transform=transform)
        self.imgs = self.samples

    def find_classes(self, directory):
        return self.classes, self.subclass_to_idx


class ImageNetVariant(MultipleDomainDataset):
    def __init__(self, root, environments, test_envs, augment, hparams,
                 classes=None, subclass_to_idx=None):
        super().__init__()

        transform = get_transform()
        augment_scheme = hparams.get('data_augmentation_scheme', 'default')
        augment_transform = get_augment_transform(augment_scheme)

        self.datasets = []
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            path = Path(root, environment)
            if classes is None:
                env_dataset = ImageFolder(path, transform=env_transform)
            else:
                env_dataset = CustomLabelMappingDataset(
                    path, classes, subclass_to_idx, transform=env_transform)
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class ImageNet_A(ImageNetVariant):
    ENVIRONMENTS = ['imagenet_train', 'imagenet_a']
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['imagenet-subset-a200/train', 'imagenet-a'],
                         test_envs, hparams['data_augmentation'], hparams)


class ImageNet_R(ImageNetVariant):
    ENVIRONMENTS = ['imagenet_train', 'imagenet_r']
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['imagenet-subset-r200/train', 'imagenet-r'],
                         test_envs, hparams['data_augmentation'], hparams)


class ImageNet_V2(ImageNetVariant):
    ENVIRONMENTS = ['imagenet_train', 'imagenet_v2']
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, ['ILSVRC/Data/CLS-LOC/train',
                                'imagenetv2-matched-frequency-format-val'],
                         test_envs, hparams['data_augmentation'], hparams)


class ImageNet_V2_Super400(ImageNetVariant):
    ENVIRONMENTS = ['imagenet_train', 'imagenet_v2']
    def __init__(self, root, test_envs, hparams):
        with Path(root, 'imagenetv2_super400_class_wnid_and_names.json').open() as f:
            classes = json.load(f)
        with Path(root, 'imagenetv2_super400_label_mapping.json').open() as f:
            subclass_to_idx = json.load(f)
        super().__init__(root, ['ILSVRC/Data/CLS-LOC/train',
                                'imagenetv2-matched-frequency-format-val'],
                         test_envs, hparams['data_augmentation'], hparams,
                         classes=classes, subclass_to_idx=subclass_to_idx)
