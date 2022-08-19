import sys
import os
from Solvers.OT_Discrete import OT_Discrete
import torch as torch
import numpy as np
import os


class Pre_Model:
    def __init__(self, cfg_proj, cfg_m):
        self.cfg_proj = cfg_proj
        self.cfg_m = cfg_m
        self.build_defense()

    def build_defense(self):
        solver_class = getattr(sys.modules[__name__], self.cfg_proj.solver) 
        self.solver = solver_class(self.cfg_proj, self.cfg_m)

    def train(self, dataloader_train, dataloader_valid):
        self.solver.train(dataloader_train, dataloader_valid)
