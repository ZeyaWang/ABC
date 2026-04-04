import numpy as np
import pandas as pd
import torch
from data import *
from net import *
from utils import *
from tqdm import tqdm
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydl import OptimizerManager, OptimWithSheduler, inverseDecaySheduler
from mixture import BayesianGaussianMixtureMerge
import sys, os
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import pickle as pk

cudnn.benchmark = True
cudnn.deterministic = True
output_device = torch.device('cuda')




Cluster = BayesianGaussianMixtureMerge(
    n_components=args.max_k,
    n_init=5,
    weight_concentration_prior=args.alpha / args.max_k,
    weight_concentration_prior_type='dirichlet_process',
    covariance_prior=args.covariance_prior * args.bottle_neck_dim * np.identity(
        args.bottle_neck_dim),
    covariance_type='full')


class OptSets():
    def __init__(self, totalNet, lr, min_step, lr_scale=0.1):
        scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=min_step)
        self.optimizer_extractor = OptimWithSheduler(
            optim.SGD(totalNet.feature_extractor.parameters(), lr=lr * lr_scale,
                      weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True), scheduler)
        self.optimizer_linear = OptimWithSheduler(
            optim.SGD(totalNet.classifier.parameters(), lr=lr , weight_decay=args.weight_decay,
                      momentum=args.momentum, nesterov=True),
            scheduler)
        self.optimizer_bottleneck = OptimWithSheduler(
            optim.SGD(totalNet.bottle_neck.parameters(), lr=lr * lr_scale, weight_decay=args.weight_decay,
                      momentum=args.momentum, nesterov=True),
            scheduler)



def detect(totalNet):
    feature_list, label_list, pred_logits = [], [], []
    with TrainingModeManager(
            [totalNet.feature_extractor, totalNet.bottle_neck, totalNet.classifier], train=False) as mgr, \
            torch.no_grad():
        for _, (im, label) in enumerate(tqdm(target_test_dl)):
            im = im.to(output_device)
            _, feature, predict_logit = totalNet(im)
            feature_list.append(feature.detach().cpu().numpy())
            pred_logits.append(predict_logit.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
    feature = np.concatenate(feature_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    sim_bmm_model = sim_bmm(norm=True)
    init_centers = totalNet.classifier.fc.weight.detach().cpu().numpy()
    cos = cosine_similarity(feature, init_centers) 
    cos_max = cos.max(1)
    cos_argmax = cos.argmax(1)
    sim_bmm_model.bmm_fit(1-cos_max)
    w_k_posterior, _ = sim_bmm_model.get_posterior(1-cos_max)
    return labels, cos_max, w_k_posterior, cos_argmax, feature


def clustering(tgt_embedding, tgt_member, predict_src):
    num_predict_src_unk = np.sum(predict_src == args.max_k)
    predict_src_known = predict_src[predict_src < args.max_k]
    uq_pred_src= np.unique(predict_src_known)
    num_uq_pred_src = len(uq_pred_src)
    n_components = num_uq_pred_src + num_predict_src_unk
    n_components = min(n_components, args.max_k)
    
    # Cluster = BayesianGaussianMixtureMerge(
    #     n_components=n_components,
    #     n_init=5,
    #     weight_concentration_prior=args.alpha / args.max_k,
    #     weight_concentration_prior_type='dirichlet_process',
    #     init_params='kmeans_merge',
    #     covariance_prior=args.covariance_prior * args.bottle_neck_dim * np.identity(
    #         args.bottle_neck_dim),
    #     covariance_type='full')

    tgt_predict = merge_cluster(Cluster, tgt_embedding, tgt_member, predict_src, num_src_cls=num_src_cls)
    metrics = {}
    return tgt_predict, metrics

def generate_memory(tgt_predict, embedding):
    tgt_predict_post, tgt_match = post_match(tgt_predict)
    memory = Memory(len(np.unique(tgt_predict_post)), feat_dim=args.bottle_neck_dim)
    memory.init(embedding, tgt_predict_post, output_device)
    return memory, tgt_predict_post, tgt_match

def train(ClustNet, train_ds, memory, optSets, epoch_step, global_step, total_step):
    num_sample = len(train_ds)
    score_bank = torch.randn(num_sample, num_src_cls).to(output_device)
    loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=3, drop_last=False)
    feat_bank = torch.randn(num_sample, args.bottle_neck_dim)
    ClustNet.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            inputs, (indx, _) = next(iter_test)
            _, feat_outputs, logit_outputs = ClustNet(inputs.to(output_device))
            feat_bank[indx] = F.normalize(feat_outputs).detach().clone().cpu()
            score_bank[indx] = nn.Softmax(-1)(logit_outputs).detach().clone()


    ClustNet.train()
    mloss_total_t, closs_total_t, loss_total_t = [], [], []
    iters = tqdm(loader, desc=f'epoch {epoch_step}', total=len(loader))
    mloss_total, closs_total, loss_total =  0., 0., 0.
    for _, (im, (idx, plabel)) in enumerate(iters):
        idx = idx.to(output_device)  
        plabel = plabel.to(output_device)
        im = im.to(output_device)
        _, feature, pred_logit = ClustNet(im)
        pred_prob = nn.Softmax(dim=1)(pred_logit)
        with torch.no_grad():
            output_feat = F.normalize(feature).cpu().detach().clone()
            feat_bank[idx] = output_feat.detach().clone().cpu()
            score_bank[idx] = pred_prob.detach().clone()
            cosine_distance = output_feat @ feat_bank.T
            _, idx_Knn = torch.topk(cosine_distance, dim=-1, largest=True, k=args.KK + 1)
            idx_Knn = idx_Knn[:, 1:]  
            score_Knn = score_bank[idx_Knn]  
        pred_prob_expand = pred_prob.unsqueeze(1).expand(-1, args.KK, -1)  
        closs = torch.mean((F.kl_div(pred_prob_expand.log(), score_Knn, reduction="none").sum(-1)).sum(1))
        mloss = memory.forward(feature, plabel)
        mloss_total += mloss.item()
        mloss = mloss * ExpWeight(global_step, max_iter=total_step*len(loader))
        loss = args.balance*closs  + mloss
        closs_total += closs.item()
        loss_total += loss.item()
        optims = [optSets.optimizer_extractor, optSets.optimizer_bottleneck, optSets.optimizer_linear]
        with OptimizerManager(optims):
            loss.backward()
        global_step += 1
        tqdm.write(f'EPOCH {epoch_step:03d}: STEP {global_step:03d}: closs={closs_total:.4f}, mloss={mloss_total:.4f}, loss={loss_total:.4f}')
        closs_total_t.append(closs_total)
        mloss_total_t.append(mloss_total)
        loss_total_t.append(loss_total)
    return global_step, closs_total_t, mloss_total_t, loss_total_t



pretrain_file = os.path.join('pretrained_source/{}/{}_{}.pkl'.format(args.target_type, args.dataset, domain_map[args.dataset][args.source]))

totalNet = SimpleNet(num_cls=num_src_cls, output_device=output_device,
                    bottle_neck_dim=args.bottle_neck_dim)

totalNet.load_model(pretrain_file, load=('feature_extractor', 'bottleneck', 'classifier'))
optSets = OptSets(totalNet, args.lr, args.total_epoch * len(target_train_dl), lr_scale=args.lr_scale)
global_step = 0
best_hos, best_acc = 0., 0.


if args.thresh is not None:
    print('Threshold is set to {}\n'.format(args.thresh))
    threshs = [args.thresh]
else:
    if args.dataset != 'visda':
        threshs = [0.4, 0.45, 0.5, 0.55, 0.6]
    else:
        threshs = [0.5]

for epoch_id in tqdm(range(args.total_epoch), desc="Processing"):

    d_result, dt = {}, {}
    tgt_member, cos_max, w_k_posterior, arg_y, tgt_embedding = detect(totalNet)

    for it, t in enumerate(threshs):
        predict_y = np.copy(arg_y)
        predict_y[w_k_posterior <= t] = args.max_k
        if (len(predict_y[predict_y < args.max_k]) == 0) or (len(predict_y[predict_y == args.max_k]) == 0):
            continue
        tgt_predict, metrics = clustering(tgt_embedding, tgt_member, predict_y)
        if len(threshs) == 1:
            sil = 1.0 
        else:
            sil = silhouette_score(tgt_embedding, tgt_predict)
        d_result[sil] = (t, tgt_predict, metrics, predict_y)
        dt[t] = sil
    max_sil = max(d_result.keys())
    t, tgt_predict, metrics, predict_y = d_result[max_sil]


    memory, tgt_predict_post, tgt_match = generate_memory(tgt_predict, tgt_embedding)
    target_train_ds.labels = list(zip([i for i in range(len(target_train_ds.datas))], tgt_predict_post))
    
    (_, unknown_test_truth, unknown_test_pred, acc_all, acc_v, hos,
    nmi_v, unk_nmi, k_acc, tgt_member, tgt_predict) = inference(memory.memory,
                                                            totalNet,
                                                            target_test_dl,
                                                            output_device,
                                                            source_classes,
                                                            tgt_match)
    metrics['nmi'] = nmi_v
    metrics['unk_nmi'] = unk_nmi
    metrics['hos'] = hos.item()
    metrics['acc'] = acc_v.item()
    metrics['tgt_member'] = tgt_member
    metrics['tgt_predict'] = tgt_predict
    metrics['epoch_id'] = epoch_id

    global_step, closs_total_t, mloss_total_t, loss_total_t = train(totalNet, target_train_ds, memory, optSets, epoch_id, global_step, args.total_epoch)

    if (epoch_id == 0) or (hos > best_hos):
        best_hos = hos
        best_metrics = metrics

print(best_metrics)
print(metrics)