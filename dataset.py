import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

Rc = 130
seed = 0
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)    
random.seed(seed)
np.random.seed(seed)

class TRANSFORMERDATA(Dataset):
    """
    root = new | old
    """
    def __init__(self, name, seq_len, root='new') -> None:
        super().__init__()
        data_root = "CMAPSS/units/"
        if root == 'old':
            label_root = "CMAPSS/labels/"
        elif root == 'new':
            label_root = "CMAPSS/new_labels/"
        else:
            raise RuntimeError("got invalid parameter root='{}'".format(root))
        raw = np.loadtxt(data_root+name)[:,2:]
        lbl = np.loadtxt(label_root+name)/Rc
        l = len(lbl)
        if l<seq_len:
            raise RuntimeError("seq_len {} is too big for file '{}' with length {}".format(seq_len, name, l))
        raw, lbl = torch.tensor(raw, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)
        lbl_pad_0 = [torch.ones([seq_len-i-1]) for i in range(seq_len-1)] 
        data_pad_0 = [torch.zeros([seq_len-i-1,24]) for i in range(seq_len-1)]
        lbl_pad_1 = [torch.zeros([i+1]) for i in range(seq_len-1)] 
        data_pad_1 = [torch.zeros([i+1,24]) for i in range(seq_len-1)]
        self.data = [torch.cat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
        self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.data += [torch.cat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
        self.label = [torch.cat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
        self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.label += [torch.cat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
        self.padding = [torch.cat([torch.ones(seq_len-i-1), torch.zeros(i+1)],0) for i in range(seq_len-1)]   # 1 for ingore
        self.padding += [torch.zeros(seq_len) for i in range(seq_len-1, l)]
        self.padding += [torch.cat([torch.zeros(seq_len-i-1), torch.ones(i+1)],0) for i in range(seq_len-1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]


class TRANSFORMER_ALL_DATA(Dataset):
    """
    root: new | old
    name: LIST of txt files to collect 
    """
    def __init__(self, name, seq_len) -> None:
        super().__init__()
        data_root = "CMAPSS/units/"
        label_root = "CMAPSS/new_labels/"
        lis = os.listdir(data_root)
        data_list = [i for i in lis if i in name]
        self.data, self.label, self.padding = [], [], []
        for n in data_list:
            raw = np.loadtxt(data_root+n)[:,2:]
            lbl = np.loadtxt(label_root+n)/Rc
            l = len(lbl)
            if l<seq_len:
                raise RuntimeError("seq_len {} is too big for file '{}' with length {}".format(seq_len, n, l))
            raw, lbl = torch.tensor(raw, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)
            lbl_pad_0 = [torch.ones([seq_len-i-1]) for i in range(seq_len-1)] 
            data_pad_0 = [torch.zeros([seq_len-i-1,24]) for i in range(seq_len-1)]
            lbl_pad_1 = [torch.zeros([i+1]) for i in range(seq_len-1)] 
            data_pad_1 = [torch.zeros([i+1,24]) for i in range(seq_len-1)]
            self.data += [torch.cat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
            self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.data += [torch.cat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
            self.label += [torch.cat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
            self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.label += [torch.cat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
            self.padding += [torch.cat([torch.ones(seq_len-i-1), torch.zeros(i+1)],0) for i in range(seq_len-1)]   # 1 for ingore
            self.padding += [torch.zeros(seq_len) for i in range(seq_len-1, l)]
            self.padding += [torch.cat([torch.zeros(seq_len-i-1), torch.ones(i+1)],0) for i in range(seq_len-1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]


class N_CMAPSS(Dataset):
    def __init__(self, name="DS01", type="train", seq_len=50, Rc=65) -> None:
        """
        FOR TRAINING !!!
        
        processes "N-CMAPSS/name/type/*.txt"
        """

        super().__init__()
        root = "N-CMAPSS/"+name+"/"+type+"/" 
        units = os.listdir(root)
        units = [i for i in units if i[-4:]==".txt"]
        self.data, self.label, self.padding = [], [], []
        for i in units:
            rw = np.loadtxt(root+i)
            data_len = len(rw)
            max_rul = int(data_len/2)
            lb = np.zeros(data_len)
            lb[0::2] = np.arange(max_rul-1, -1, -1)
            lb[1::2] = np.arange(max_rul-1, -1, -1)
            for k in range(len(lb)):
                if lb[k] > Rc:
                    lb[k] = Rc
            lb = lb/Rc
            rw = np.concatenate([rw, np.zeros([rw.shape[0],2])], 1)
            raw, lbl = torch.tensor(rw, dtype=torch.float), torch.tensor(lb, dtype=torch.float)
            l = len(lb)
            lbl_pad_0 = [torch.ones([seq_len-i-1]) for i in range(seq_len-1)] 
            data_pad_0 = [torch.zeros([seq_len-i-1,40]) for i in range(seq_len-1)]
            lbl_pad_1 = [torch.zeros([i+1]) for i in range(seq_len-1)] 
            data_pad_1 = [torch.zeros([i+1,40]) for i in range(seq_len-1)]
            self.data += [torch.cat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
            self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.data += [torch.cat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
            self.label += [torch.cat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
            self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
            self.label += [torch.cat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
            self.padding += [torch.cat([torch.ones(seq_len-i-1), torch.zeros(i+1)],0) for i in range(seq_len-1)]   # 1 for ingore
            self.padding += [torch.zeros(seq_len) for i in range(seq_len-1, l)]
            self.padding += [torch.cat([torch.zeros(seq_len-i-1), torch.ones(i+1)],0) for i in range(seq_len-1)]
            
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]


class N_CMAPSS_TEST(Dataset):
    def __init__(self, domain="DS01", type="test", name="unit1.txt", seq_len=50, Rc=65) -> None:
        """
        FOR TESTING !!!

        process "N-CMAPSS/domain/type/name"
        
        """
        
        root = "N-CMAPSS/"+domain+"/"+type+"/"
        super().__init__()
        self.data, self.label, self.padding = [], [], []
        rw = np.loadtxt(root+name)
        data_len = len(rw)
        max_rul = int(data_len/2)
        lb = np.zeros(data_len)
        lb[0::2] = np.arange(max_rul-1, -1, -1)
        lb[1::2] = np.arange(max_rul-1, -1, -1)
        for k in range(len(lb)):
            if lb[k] > Rc:
                lb[k] = Rc
        lb = lb/Rc
        rw = np.concatenate([rw, np.zeros([rw.shape[0],2])], 1)
        raw, lbl = torch.tensor(rw, dtype=torch.float), torch.tensor(lb, dtype=torch.float)
        l = len(lb)
        lbl_pad_0 = [torch.ones([seq_len-i-1]) for i in range(seq_len-1)] 
        data_pad_0 = [torch.zeros([seq_len-i-1,40]) for i in range(seq_len-1)]
        lbl_pad_1 = [torch.zeros([i+1]) for i in range(seq_len-1)] 
        data_pad_1 = [torch.zeros([i+1,40]) for i in range(seq_len-1)]
        self.data += [torch.cat([data_pad_0[i],raw[:i+1]],0) for i in range(seq_len-1)] 
        self.data += [raw[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.data += [torch.cat([raw[l-seq_len+i+1:], data_pad_1[i]],0) for i in range(seq_len-1)]
        self.label += [torch.cat([lbl_pad_0[i],lbl[:i+1]],0) for i in range(seq_len-1)] 
        self.label += [lbl[i-seq_len+1:i+1] for i in range(seq_len-1, l)]
        self.label += [torch.cat([lbl[l-seq_len+i+1:], lbl_pad_1[i]],0) for i in range(seq_len-1)]
        self.padding += [torch.cat([torch.ones(seq_len-i-1), torch.zeros(i+1)],0) for i in range(seq_len-1)]   # 1 for ingore
        self.padding += [torch.zeros(seq_len) for i in range(seq_len-1, l)]
        self.padding += [torch.cat([torch.zeros(seq_len-i-1), torch.ones(i+1)],0) for i in range(seq_len-1)]
            
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]


if __name__ == "__main__":
    pass
    