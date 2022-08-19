from Data.dataset_class import MNIST, CIFAR10_PAIR
import sys
from torch.utils.data import DataLoader, random_split


def pre_data(data_name, cfg_proj, cfg_m):

    #pre the dataset according to the data_name
    dataset_class = getattr(sys.modules[__name__], data_name.upper()) 

    #split the training set into train and valid
    dataset_train = dataset_class(flag_train = True, cfg_m = cfg_m)
    if cfg_m.valid_rate > 0:
        rate = [int(len(dataset_train)*(1-cfg_m.valid_rate)), int(len(dataset_train)*cfg_m.valid_rate)]        
        rate[0] = rate[0] + (len(dataset_train) - sum(rate))  
        dataset_train, dataset_valid = random_split(dataset_train, rate)
        dataloader_valid= DataLoader(dataset=dataset_valid, batch_size=cfg_m.batch_size, num_workers=8, drop_last=True, shuffle=False)
    else:
        dataset_valid = None
        dataloader_valid = None
    dataloader_train= DataLoader(dataset=dataset_train, batch_size=cfg_m.batch_size, num_workers = 16, drop_last=True, shuffle=True)
    
    #test data
    if data_name.upper() == "STL10_UNLABELED":
        dataset_test = None
        dataloader_test = None
    else:
        dataset_test = dataset_class(flag_train = False, cfg_m = cfg_m)
        dataloader_test= DataLoader(dataset=dataset_test, batch_size=cfg_m.batch_size, num_workers = 8, drop_last=True, shuffle=False)

    return [dataloader_train, dataloader_valid, dataloader_test], [dataset_train, dataset_valid, dataset_test]