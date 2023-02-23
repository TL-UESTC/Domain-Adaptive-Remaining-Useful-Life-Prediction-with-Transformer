import torch.nn as nn
from model import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
from dataset import TRANSFORMER_ALL_DATA, TRANSFORMERDATA
from torch.utils.data import DataLoader, random_split
from loss import advLoss
import itertools
import time


def validate():
    net.eval()
    tot = 0
    with torch.no_grad():
        for i in target_test_names:
            pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
            valid_data = TRANSFORMERDATA(i, seq_len)
            data_len = len(valid_data)
            valid_loader = DataLoader(valid_data, batch_size=1000)
            valid_iter = iter(valid_loader)
            d = next(valid_iter)
            input, lbl, msk = d[0], d[1], d[2]
            input, msk = input.cuda(), msk.cuda()
            _, out = net(input, msk)
            out = out.squeeze(2).cpu()
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
            truth = torch.tensor([lbl[j,-1] for j in range(len(lbl)-seq_len+1)], dtype=torch.float)
            pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
            pred = pred_sum/pred_cnt
            mse = float(torch.sum(torch.pow(pred-truth, 2)))
            rmse = math.sqrt(mse/data_len)
            tot += rmse
    return tot*Rc/len(valid_list)


def train():
    minn = 999
    for e in range(epochs):
        al, tot = 0, 0
        net.train()
        random.shuffle(source_list)
        random.shuffle(target_list)
        source_iter, target_iter = iter(source_list), iter(target_list)
        loss2_sum, loss1_sum = 0, 0
        bkb_sum, out_sum = 0, 0
        cnt = 0
        s_iter = iter(DataLoader(s_data, batch_size=args.batch_size, shuffle=True))
        t_iter = iter(DataLoader(t_data, batch_size=args.batch_size, shuffle=True))
        l = min(len(s_iter), len(t_iter))
        for _ in range(l):
            s_d, t_d = next(s_iter), next(t_iter)
            s_input, s_lb, s_msk = s_d[0], s_d[1], s_d[2]
            t_input, t_msk = t_d[0], t_d[2]
            s_input, s_lb, s_msk = s_input.cuda(), s_lb.cuda(), s_msk.cuda()
            t_input, t_msk = t_input.cuda(), t_msk.cuda()
            s_features, s_out = net(s_input, s_msk)
            t_features, t_out = net(t_input, t_msk) # [bts, seq_len, feature_num]
            s_out.squeeze_(2)
            t_out.squeeze_(2)
            loss1 = Loss(s_out, s_lb)
            loss1_sum += loss1
            cnt += 1
            if args.type == 1 or args.type == 0:
                if args.type == 1:
                    s_domain = D2(s_features)
                    t_domain = D2(t_features)
                else:
                    s_domain = D1(s_out)
                    t_domain = D1(t_out)
                loss2 = advLoss(s_domain.squeeze(1), t_domain.squeeze(1), 'cuda')
                loss2_sum += loss2
                loss = loss1 + a*loss2
            elif args.type == 2:
                s_domain_bkb = D2(s_features)
                t_domain_bkb = D2(t_features)
                s_domain_out = D1(s_out)
                t_domain_out = D1(t_out)
                if e>=5:
                    fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1), 'cuda')
                    out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), 'cuda')
                    bkb_sum += fea_loss
                    out_sum += out_loss
                    loss = loss1 + a*fea_loss + b*out_loss
                else:
                    loss = loss1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), 2)
            opt.step()    

        rmse = validate()
        if args.type == 2:
            print("{}/{}| loss1={:.5f}, fea_loss={:.5f}, out_loss={:.5f}, rmse={:.5f}".\
                format(e, args.epoch, loss1_sum/cnt, bkb_sum/cnt, out_sum/cnt, rmse))
        else:    
            print("{}/{}| 1={:.5f}, 2={:.5f}, rmse={:.5f}".format(e, args.epoch, loss1, loss2_sum/cnt, rmse))
        if rmse<minn:
            minn = rmse
            print("min={}".format(minn))
            if args.type == 1:
                torch.save(net.state_dict(), "save/final/dann_"+source[-1]+target[-1]+".pth")
            elif args.type == 0:
                torch.save(net.state_dict(), "save/final/out_"+source[-1]+target[-1]+".pth")
            elif args.type == 2 :
                #torch.save(net.state_dict(), "save/final/both_"+source[-1]+target[-1]+".pth")
                torch.save(net.state_dict(), "online/"+source[-1]+target[-1]+"_net.pth")
                torch.save(D1.state_dict(), "online/"+source[-1]+target[-1]+"_D1.pth")
                torch.save(D2.state_dict(), "online/"+source[-1]+target[-1]+"_D2.pth")
        
        if args.scheduler:
            sch.step()

    return minn


def get_score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)    
    random.seed(seed)
    np.random.seed(seed)
    Rc = 130

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument("--epoch", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--seq_len", type=int, default=70)
    parser.add_argument("--source", type=str, default="FD003", help="decide source file", choices=['FD001','FD002','FD003','FD004'])
    parser.add_argument("--target", type=str, default="FD002", help="decide target file", choices=['FD001','FD002','FD003','FD004'])
    parser.add_argument("--a", type=float, default=0.1, help='hyper-param α')
    parser.add_argument("--b", type=float, default=0.5, help='hyper-param β')
    parser.add_argument("--scheduler", type=int, default=1, choices=[0,1], help="1 for using sheduler while 0 for not")
    parser.add_argument("--type", type=int, default=2, choices=[0,1,2], help="0:out only | 1:DANN | 2:backbone+output")
    parser.add_argument("--train", default=1, type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    source, target = args.source, args.target
    data_root = "CMAPSS/units/"
    label_root = "CMAPSS/labels/"
    type = {0:"out_only", 1:"DANN", 2:"backbone + output"}
    seq_len, a, epochs, b = args.seq_len, args.a, args.epoch, args.b
    option_str = "source={}, target={}, a={}, b={}, epochs={}, type={}, lr={}, {}using scheduler".\
        format(source, target, a, b, epochs, type[args.type], args.lr, "" if args.scheduler else "not ")
    print(option_str)

    net = mymodel(max_len=seq_len) 
    D1 = Discriminator(seq_len)
    D2 = backboneDiscriminator(seq_len)
    if args.type == 0:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters()), lr=args.lr)
    elif args.type == 1:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D2.parameters()), lr=args.lr)
    elif args.type == 2:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), lr=args.lr)
    Loss = nn.MSELoss()
    net, Loss, D1, D2 = net.cuda(), Loss.cuda(), D1.cuda(), D2.cuda()
    sch = torch.optim.lr_scheduler.StepLR(opt, 80, 0.5)

    source_list = np.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
    target_list = np.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
    valid_list = np.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
    a_list = np.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
    target_test_names = valid_list + a_list
    minl = min(len(source_list), len(target_list))
    s_data = TRANSFORMER_ALL_DATA(source_list, seq_len)
    t_data = TRANSFORMER_ALL_DATA(target_list, seq_len)
    t_data_test = TRANSFORMER_ALL_DATA(target_test_names, seq_len)
    if not os.path.exists('./online'):
        os.makedirs('./online')

    if args.train:
        train_time1 = time.perf_counter()
        minn = train()
        train_time2 = time.perf_counter()
        print(option_str)
        print("best = {}, train time = {}".format(minn, train_time2-train_time1))


