#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:26:42 2018

@author: gaoyi
"""

from .mnist import get_mnist
from .mnist_m import get_mnist_m

__all__ = (get_mnist_m, get_mnist)
