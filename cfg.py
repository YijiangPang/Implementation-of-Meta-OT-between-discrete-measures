import os
from Utils.utils import dotdict



def init_cfg(n_solver):
    cfg_m = dotdict()
    cfg_m.insert("Note", None)
    cfg_m.insert("datasets_root", '../datasets')

    if n_solver == "OT_Discrete":
        cfg_m.insert("epochs", 10)
        cfg_m.insert("learning_rate_init", 1e-3)
        cfg_m.insert("log_interval", 100)
        cfg_m.insert("batch_size", 1024)
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("img_size", 28)
        cfg_m.insert("epsilon", 1e-2)
        cfg_m.insert("MLP_hidden_num", 3)

    return cfg_m