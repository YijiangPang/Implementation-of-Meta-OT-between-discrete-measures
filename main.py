import os
import argparse
from Data.pre_data import pre_data
from Models.pre_model import Pre_Model
from time import localtime, strftime
from cfg import init_cfg


def main(cfg_proj, cfg_m):
    model = Pre_Model(cfg_proj, cfg_m)
    [dataloader_train, dataloader_valid, dataloader_test], [_, _, _] = pre_data(cfg_proj.data_name, cfg_proj, cfg_m)
    model.train(dataloader_train, dataloader_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OT")
    parser.add_argument("--gpu", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default = 1, required=False)   #np.random.randint(0, 1000000)
    #proj cfg
    parser.add_argument("--data_name", type=str, default="MNIST", required=False)   #MNIST, CIFAR10_PAIR
    parser.add_argument("--solver", type=str, default="OT_Discrete", required=False)  #OT_Discrete, 
    parser.add_argument("--flag_time", type=str, default = strftime("%Y-%m-%d_%H-%M-%S", localtime()), required=False)
    parser.add_argument("--flag_load", type=str, default = None, required=False)    #if is not None, then the file of loaded para need to contain the str
    cfg_proj = parser.parse_args()

    cfg_m = init_cfg(cfg_proj.solver)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(cfg_proj.gpu)
    main(cfg_proj, cfg_m)