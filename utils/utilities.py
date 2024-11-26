import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch import linalg as LA

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.decomposition import PCA

def save_np(filename,nump,path):
    np.save(path+filename,nump)

def model_ood_detection(model,id_dataloader,oe_dataloader,path,args):
    pred = {"id_pred":[],"oe_pred":[],"mix_pred":[]}
    l2_pred = {"id_pred":[],"oe_pred":[],"mix_pred":[]}
    logits = {"id_logit":[],"oe_logit":[],"mix_logit":[]}
    labels = {"id_label":[],"oe_label":[],"mix_label":[]}

    lam = np.random.beta(args.alpha, args.alpha)
    model.eval()

    with torch.no_grad():
        for data in id_dataloader:
            inputs, label = data[0].to("cuda"), data[1].to("cuda")
            output = model(inputs)  # モデルから予測を計算(順伝播計算)：tensor(BATCH_SIZE, 確率×10)
            pred = F.softmax(output,dim=1)
            pred_l2 = LA.norm(output,dim=1)
            values, indices = torch.max(pred.data, dim=1)
            # print(f"pred.shape: {pred.shape}")  # ソフトマックス後の形状
            # print(f"values.shape: {values.shape}")  # 最大値の形状
            pred["id_pred"].append(values.to('cpu').detach().numpy())
            l2_pred["id_pred"].append(pred_l2.to('cpu').detach().numpy().copy())
            logits["id_logit"].append(output.to('cpu').detach().numpy().copy())
            labels["id_label"].append(label.to('cpu').detach().numpy().copy())

    with torch.no_grad():
        for data in oe_dataloader:
            inputs, label = data[0].to("cuda"), data[1].to("cuda")
            output = model(inputs)  # モデルから予測を計算(順伝播計算)：tensor(BATCH_SIZE, 確率×10)
            pred = F.softmax(output,dim=1)
            pred_l2 = LA.norm(output,dim=1)
            values, indices = torch.max(pred.data, dim=1)
            pred["oe_pred"].append(values.to('cpu').detach().numpy().copy())
            l2_pred["oe_pred"].append(pred_l2.to('cpu').detach().numpy().copy())
            logits["oe_logit"].append(output.to('cpu').detach().numpy().copy())
            labels["oe_label"].append(label.to('cpu').detach().numpy().copy())
    
    with torch.no_grad():
        for data1,data2 in zip(id_dataloader,oe_dataloader):
            inputs1, label1 = data1[0].to("cuda"), data1[1].to("cuda")
            inputs2, label2 = data2[0].to("cuda"), data2[1].to("cuda")
            output1 = model(inputs1)
            output2 = model(inputs2)
            mixed_output = lam * output1 + (1 - lam) * output2
            pred = F.softmax(mixed_output,dim=1)
            pred_l2 = LA.norm(mixed_output,dim=1)
            values, indices = torch.max(pred.data, dim=1)
            pred["mix_pred"].append(values.to('cpu').detach().numpy().copy())
            l2_pred["mix_pred"].append(pred_l2.to('cpu').detach().numpy().copy())
            logits["mix_logit"].append(output.to('cpu').detach().numpy().copy())
            labels["mix_label"].append(label.to('cpu').detach().numpy().copy())

    ood_detection_score(pred,l2_pred,path)
    pca_plt(logits,labels,path)
    
    

def ood_detection_score(val_sof,val_l2_norm,path):
    id_soft_pred = list(itertools.chain.from_iterable(val_sof["id_pred"]))
    oe_soft_pred = list(itertools.chain.from_iterable(val_sof["oe_pred"]))
    id_l2_pred = list(itertools.chain.from_iterable(val_l2_norm["id_pred"]))
    oe_l2_pred = list(itertools.chain.from_iterable(val_l2_norm["oe_pred"]))

    plt.clf()

    plt.subplot(1,2,1)
    plt.hist(id_soft_pred,bins = 200,alpha = 0.6,label="ID")
    plt.hist(oe_soft_ored,bins = 200,alpha = 0.4,label="OOD")
    plt.xlim(0,1)
    plt.ylim(0,10000)
    plt.title("softmax")
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(id_l2_pred,bins = 200,alpha = 0.6,label="ID")
    plt.hist(oe_l2_pred,bins = 200,alpha = 0.4,label="OOD")
    plt.xlim(0,50)
    plt.ylim(0,1000)
    plt.title("L2 normalization")
    plt.legend()
    plt.show()

    plt.savefig(path + "ood_detection.png")

    soft_pred = id_soft_pred + oe_soft_pred
    soft_true = np.array([1]*len(id_soft_pred) + [0]*len(oe_soft_pred))
    unique_classes = np.unique(soft_true)

    l2_pred = id_l2_pred + oe_l2_pred
    l2_true = np.array([1]*len(id_l2_pred) + [0]*len(oe_l2_pred))
    unique_classes = np.unique(l2_true)

    roc_soft = roc_curve(soft_true,soft_pred)
    roc_l2 = roc_curve(l2_true,l2_pred)

    auc_soft = roc_auc_score(soft_true,soft_pred)
    auc_l2 = roc_auc_score(l2_true,l2_pred)

    plt.clf()
    plt.figure(1, figsize=(13,4))
    plt.plot(roc_soft[0],roc_soft[1],label="softmax")
    plt.plot(roc_l2[0],roc_l2[1],label="L2 Normalization")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.legend()
    plt.text(0,0,"AUC score : softmax {:.4f}  L2 norm {:.4f}".format(auc_soft,auc_l2))

    plt.savefig(path+"AUROC.png")

    print(f"AUC score : softmax {auc_soft:.4f}  L2 norm {auc_l2:.4f}")

def pca_plot(logit,labels,path):
    pca = PCA(n_components=2)

    id_logit = np.concatenate(logit['id_logit'], axis=0)
    oe_logit = np.concatenate(logit['oe_logit'], axis=0)
    mix_logit = np.concatenate(logit['mix_logit'], axis=0)
    id_label = np.concatenate(labels['id_label'], axis=0)
    oe_label = np.concatenate(labels['oe_label'], axis=0)
    mix_label = np.concatenate(labels['mix_label'], axis=0)

    x = pca.fit_transform(np.concatenate([id_logit, oe_logit, mix_logit], axis=0))
    id_x = x[:len(id_logit)]
    oe_x = x[len(id_logit):len(oe_logit)]
    mix_x = x[len(oe_logit):]

    plt.clf()
    plt.figure(1, figsize=(13,4))
    plt.scatter(id_x[:1000,0],id_x[:1000,1],label="ID",color="r",marker=".",alpha=0.3)
    plt.scatter(oe_x[:1000,0],oe_x[:1000,1],label="OOD",color="b",marker=".",alpha=0.3)
    plt.scatter(mix_x[:1000,0],mix_x[:1000,1],label="Mix",color="g",marker=".",alpha=0.3)
    plt.legend()
    plt.savefig(path+"pca.png")