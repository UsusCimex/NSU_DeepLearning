"""Мини-библиотека для многослойного перцептрона на numpy по образцу PyTorch."""

from . import data, metrics, optim
from .activations import ReLU, Sigmoid, Tanh
from .layers import Linear, Sequential
from .losses import CrossEntropyLoss, MSELoss
from .module import Module, Parameter
from .random import manual_seed
