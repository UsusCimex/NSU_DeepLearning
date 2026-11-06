"""Мини-библиотека для многослойного перцептрона на numpy по образцу PyTorch."""

from . import metrics, optim
from .activations import Tanh
from .layers import Linear, Sequential
from .losses import MSELoss
from .module import Module, Parameter
from .random import manual_seed
