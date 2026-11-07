"""Выборка изображений MNIST по образцу torchvision.datasets.
У выборки data - яркости 0..255, targets - номера классов. Файлы ищутся в root,
при download=True недостающие скачиваются. TensorDataset и DataLoader делят выборку
на батчи для обучения."""

import gzip
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .random import generator


def download_file(urls, path):
    """Скачивает файл path.name по первому рабочему адресу из urls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')  # недокачанный файл не примется за готовый
    for url in urls:
        try:
            urllib.request.urlretrieve(url + path.name, part)
        except OSError:
            continue
        part.replace(path)
        return path
    raise RuntimeError(f'не удалось скачать {path.name}')


def read_idx(path):
    """Файл IDX: магическое число, размеры по осям, затем байты данных."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        # '>I' - целое без знака из 4 байт, старший байт первым;
        # младший байт магического числа - число осей
        magic, = struct.unpack('>I', f.read(4))
        n_dims = magic & 0xff
        dims = struct.unpack('>' + 'I' * n_dims, f.read(4 * n_dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


class MNIST:
    """Рукописные цифры: 60000 обучающих (train=True) и 10000 тестовых изображений.
    data имеет форму (n, 28, 28)."""

    URLS = ['https://ossci-datasets.s3.amazonaws.com/mnist/',
            'https://storage.googleapis.com/cvdf-datasets/mnist/']
    FILES = {True: ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
             False: ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')}

    def __init__(self, root, train=True, download=True):
        images, labels = self.FILES[train]
        self.data = read_idx(self._file(Path(root), images, download))
        self.targets = read_idx(self._file(Path(root), labels, download)).astype(np.int64)

    def _file(self, root, name, download):
        for path in (root / name, root / (name + '.gz')):
            if path.exists():
                return path
        if not download:
            raise FileNotFoundError(root / name)
        return download_file(self.URLS, root / (name + '.gz'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class TensorDataset:
    """Выборка из массивов одинаковой длины: объект i - строки i всех массивов.
    Индекс может быть числом, срезом или массивом номеров."""

    def __init__(self, *arrays):
        if any(len(a) != len(arrays[0]) for a in arrays):
            raise ValueError('массивы разной длины')
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, index):
        return tuple(a[index] for a in self.arrays)


class DataLoader:
    """Батчи выборки по batch_size объектов; при shuffle в каждой эпохе новый порядок."""

    def __init__(self, dataset, batch_size=64, shuffle=False):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle

    def __iter__(self):
        n = len(self.dataset)
        order = generator().permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            yield self.dataset[order[start:start + self.batch_size]]

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
