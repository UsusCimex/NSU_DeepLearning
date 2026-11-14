"""Выборка изображений MNIST по образцу torchvision.datasets.
У выборки data - яркости 0..255, targets - номера классов. Файлы ищутся в root,
при download=True недостающие скачиваются."""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


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
    if path.suffix == '.gz':
        f = gzip.GzipFile(path, 'rb')
    else:
        f = open(path, 'rb')
    with f:
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
