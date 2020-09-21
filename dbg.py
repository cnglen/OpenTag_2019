#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""

__author__ = ""
__email__ = ""
__status__ = "DEV"

import torch.nn as nn
import torch
from typing import Dict, List, Tuple
import logging
import models
from config import opt


def train(**kwargs):
    # d = torch.device("cuda")
    # m = nn.Linear(100, 3).to(d)
    opt._parse(kwargs)
    model = getattr(models, opt.model)(opt)
    model.to(opt.device)        # print(opt.device)
    print("done")


if __name__ == '__main__':
    import fire
    fire.Fire()
