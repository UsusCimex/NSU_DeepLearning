"""Выборки и батчи: TensorDataset, random_split, DataLoader и загрузчик MNIST."""

import gzip
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .random import generator


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


def random_split(dataset, lengths):
    """Случайное разбиение выборки на непересекающиеся части заданных размеров."""
    if sum(lengths) != len(dataset):
        raise ValueError('сумма размеров частей не равна размеру выборки')
    order = generator().permutation(len(dataset))
    parts, start = [], 0
    for n in lengths:
        parts.append(TensorDataset(*dataset[order[start:start + n]]))
        start += n
    return parts


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


class MNIST(TensorDataset):
    """Рукописные цифры 28 x 28: 60000 обучающих (train=True) и 10000 тестовых.
    Объект - вектор из 784 яркостей в [0, 1], метка - цифра. Файлы ищутся в root,
    при download=True недостающие скачиваются."""

    MIRRORS = ['https://ossci-datasets.s3.amazonaws.com/mnist/',
               'https://storage.googleapis.com/cvdf-datasets/mnist/']
    FILES = {True: ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
             False: ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')}

    def __init__(self, root='data/MNIST', train=True, download=True):
        self.root = Path(root)
        images_file, labels_file = self.FILES[train]
        self.images = read_idx(self._find(images_file, download))  # (n, 28, 28), яркости 0..255
        self.labels = read_idx(self._find(labels_file, download)).astype(np.int64)
        pixels = self.images.reshape(len(self.images), -1).astype(np.float32) / 255.0
        super().__init__(pixels, self.labels)

    def _find(self, name, download):
        for path in (self.root / name, self.root / (name + '.gz')):
            if path.exists():
                return path
        if not download:
            raise FileNotFoundError(self.root / name)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (name + '.gz')
        part = self.root / (name + '.gz.part')  # недокачанный файл не примется за готовый
        for mirror in self.MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + path.name, part)
            except OSError:
                continue
            part.replace(path)
            return path
        raise RuntimeError(f'не удалось скачать {path.name}')
