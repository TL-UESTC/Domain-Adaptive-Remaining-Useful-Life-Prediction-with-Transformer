import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TRANSFORMERDATA
from model import *
import os
import random
from sklearn.manifold import TSNE


def score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


def get_pred_result(data_len, out, lb):
    pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
    for j in range(data_len):
        if j < seq_len-1:
            pred_sum[:j+1] += out[j, -(j+1):]
            pred_cnt[:j+1] += 1
        elif j <= data_len-seq_len:
            pred_sum[j-seq_len+1:j+1] += out[j]
            pred_cnt[j-seq_len+1:j+1] += 1
        else:
            pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += out[j, :(data_len-j)]
            pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += 1
    truth = torch.tensor([lb[j,-1] for j in range(len(lb)-seq_len+1)], dtype=torch.float)
    pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
    pred2 = pred_sum/pred_cnt
    pred2 *= Rc
    truth *= Rc
    return truth, pred2 


def test():
    truth, tot, tot_sc = [], 0, 0
    net.eval()
    s_model.eval()
    t_model.eval()
    with torch.no_grad():
        for k in range(test_len):
            i = next(list_iter)
            dataset = TRANSFORMERDATA(i, seq_len)
            data_len = len(dataset)
            dataloader = DataLoader(dataset, batch_size=800, shuffle=0)
            it = iter(dataloader)
            d = next(it)
            input, lb, msk = d[0], d[1], d[2]
            if fake:
                input = torch.zeros(input.shape)
            input, msk = input.cuda(), msk.cuda()
            #uncertainty(input, msk, data_len, lb, i)
            _, out = net(input, msk)
            out = out.squeeze(2).cpu()
            truth, pred = get_pred_result(data_len, out, lb)
            mse = float(torch.sum(torch.pow(pred-truth, 2)))
            rmse = math.sqrt(mse/data_len)
            tot += rmse
            sc = score(pred, truth)
            tot_sc += sc
            print("for file {}: rmse={:.4f}, score={}".format(i, rmse, sc))
            print('-'*80)
           
    print("tested on [{}] files, mean RMSE = {:.4f}, mean score = {}".format(test_len, tot/test_len, int(tot_sc/test_len)))


if __name__ == "__main__": 
    Rc = 130
    fake = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--seq_len", type=int, default=70)
    parser.add_argument("--source", type=str, default="FD002", help="file name the model trained on")
    parser.add_argument("--target", type=str, default="FD004", help="test domain")
    parser.add_argument("--sem", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seq_len = args.seq_len
    net = mymodel(max_len=seq_len, dropout=0.5).cuda()
    model_name = args.source
    test_name = args.target
    new = 'both_'+args.source[-1]+args.target[-1]
    x=torch.load("save/final/"+new+".pth", map_location='cuda:0')

    net.load_state_dict(x)
    data_root = "CMAPSS/units/"
    label_root = "CMAPSS/labels/"
    lis = os.listdir(data_root)
    test_list = [i for i in lis if i[:5] == test_name]
    random.shuffle(test_list)
    test_len = len(test_list)
    list_iter = iter(test_list)
    s_model, t_model = mymodel(max_len=seq_len).cuda(), mymodel(max_len=seq_len).cuda()
    s_pth = "save/final/FD00"+new[-2]+"new.pth"
    t_pth = "save/final/FD00"+new[-1]+"new.pth"
    s_model.load_state_dict(torch.load(s_pth, map_location="cuda:0"))
    t_model.load_state_dict(torch.load(t_pth, map_location="cuda:0"))
    test()
    
