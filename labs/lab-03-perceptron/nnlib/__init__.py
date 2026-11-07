"""Мини-библиотека для перцептрона на numpy по образцу PyTorch."""

from . import data, metrics, optim
from .layers import Linear
from .losses import CrossEntropyLoss
from .module import Module, Parameter
from .random import manual_seed
