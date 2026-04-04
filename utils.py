# Some functions in this module are adapted from:
# https://github.com/Solacex/Domain-Consensus-Clustering
#
# MIT License
# Copyright (c) 2021 Guangrui Li
# https://github.com/Solacex/Domain-Consensus-Clustering/blob/main/LICENSE

import numpy as np
from easydl import AccuracyCounter, TrainingModeManager
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from function import BetaMixture1D
from sklearn.metrics.cluster import contingency_matrix


def report(predict_id, label, source_classes):
    acc_counter = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    unknown_test_truth, unknown_test_pred = [], []
    known_test_truth, known_test_pred = [], []

    for (each_pred_id, each_label) in zip(predict_id, label):
        if each_label in source_classes:
            acc_counter[each_label].Ntotal += 1.0
            if each_pred_id == each_label:
                acc_counter[each_label].Ncorrect += 1.0
        else:
            acc_counter[-1].Ntotal += 1.0
            if each_pred_id >= len(source_classes):
                acc_counter[-1].Ncorrect += 1.
                
        if each_pred_id in source_classes:
            known_test_pred.append(each_pred_id)
            known_test_truth.append(each_label)
        else:
            unknown_test_pred.append(each_pred_id)
            unknown_test_truth.append(each_label)


    acc_all = {ii: x.reportAccuracy() for ii, x in enumerate(acc_counter) if not np.isnan(x.reportAccuracy())}
    return acc_counter, unknown_test_truth, unknown_test_pred, known_test_truth, known_test_pred, acc_all


class Memory(nn.Module):
    def __init__(self, num_cls=10, feat_dim=256):
        super(Memory, self).__init__()
        self.num_cls = num_cls
        self.feat_dim = feat_dim
        self.memory = torch.zeros(self.num_cls, feat_dim, dtype=torch.float).cuda()

    def init(self, embeddings, labels, device):
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        euclidean_centers = np.zeros((len(unique_classes), embeddings.shape[1]))
        for i, class_label in enumerate(unique_classes):
            class_indices = np.where(labels == class_label)[0]
            class_embeddings = embeddings[class_indices]
            class_center = np.mean(class_embeddings, axis=0)
            euclidean_centers[i] = class_center
        euclidean_centers = torch.from_numpy(euclidean_centers).float().to(device)
        self.memory = F.normalize(euclidean_centers, p=2, dim=-1)

    def update_center_by_simi(self, batch_center, flags):
        old_center = self.memory
        update_wei = (old_center * batch_center).sum(dim=-1).squeeze()
        update_wei = update_wei.view(-1, 1).expand_as(old_center)
        flags = flags.expand_as(self.memory)
        update_wei = torch.ones_like(flags) - (1 - update_wei) * flags  # update_wei
        self.memory = update_wei * self.memory + (1 - update_wei) * batch_center
        self.memory = F.normalize(self.memory, p=2, dim=-1)

    def update(self, feat, label):
        feat = feat.detach()
        batch_center = []
        empty = torch.zeros((1, self.feat_dim), dtype=torch.float).cuda()
        flags = []
        for i in range(self.num_cls):
            mask = label == i
            if mask.sum() == 0:
                flags.append(torch.Tensor([.0]).cuda())
                batch_center.append(empty)
                continue
            index = mask.squeeze().nonzero(as_tuple=False)
            cur_feat = feat[index, :]
            cur_feat = cur_feat.sum(dim=0)
            cur_feat = F.normalize(cur_feat, p=2, dim=-1)
            cur_feat = cur_feat.view(1, -1)

            flags.append(torch.Tensor([1.0]).cuda())
            batch_center.append(cur_feat)
        batch_center = torch.cat(batch_center, dim=0)
        flags = torch.stack(flags).cuda()
        self.update_center_by_simi(batch_center, flags)

    def forward(self, feat, label, t=0.1): # t = 1.0 or 0.1
        feat = F.normalize(feat, p=2, dim=-1)
        self.update(feat, label.unsqueeze(0))
        simis = torch.matmul(feat, self.memory.transpose(0, 1))
        simis = simis / t
        loss = F.cross_entropy(simis, label.squeeze())
        return loss.mean()


def post_match(t_codes):
    # make label of t_codes to be 0-index (0,1,2,...)
    # then return the match relationship between src center and tar center
    # return reindex t_codes and the map (src_index, tar_index)
    unique_elements, indices = np.unique(t_codes, return_inverse=True)
    match = dict(zip(range(len(unique_elements)), unique_elements))
    return indices, match



def cos_simi(x1, x2):
    simi = torch.matmul(x1, x2.transpose(0, 1))
    return simi


def map_values(x, mapping_dict):
    return mapping_dict.get(x, x)

# Vectorize the function
vectorized_map = np.vectorize(map_values)



def inference(t_centers , Net, target_test_dl, output_device, source_classes, tgt_match):
    tgt_member, tgt_predict = [], []
    Net.eval()
    for i, (im_target, label_target) in enumerate(target_test_dl):
        im_target = im_target.to(output_device)
        _, feature_target, _ = Net(im_target)
        tgt_member.append(label_target.detach().cpu().numpy())
        clus_index = cos_simi(F.normalize(feature_target, p=2, dim=-1), t_centers).argmax(dim=-1) 
        tgt_predict.append(clus_index.detach().cpu().numpy())

    tgt_member = np.concatenate(tgt_member, axis=0)
    tgt_predict = np.concatenate(tgt_predict, axis=0)
    tgt_predict = vectorized_map(tgt_predict, tgt_match)
    Net.train()
    acc_counter, unknown_test_truth, unknown_test_pred, known_test_truth, known_test_pred, acc_all = report(tgt_predict, tgt_member, source_classes)
    acc_v = np.round(np.mean(list(acc_all.values())), 3)
    acc_all_values = np.array(list(acc_all.values()))
    kn_acc = np.mean(acc_all_values[:-1])
    unk_acc = acc_all_values[-1]
    hos = 2 * (kn_acc * unk_acc) / (kn_acc + unk_acc)
    nmi_v = nmi(tgt_member,tgt_predict)
    unk_nmi = nmi(unknown_test_truth, unknown_test_pred)

    k_acc = np.sum(np.array(known_test_truth) == np.array(known_test_pred)) / len(known_test_truth)
    return acc_counter, unknown_test_truth, unknown_test_pred, acc_all, acc_v, hos, nmi_v, unk_nmi, k_acc, tgt_member, tgt_predict


def gen_cluster_input(Net, target_test_dl, output_device):
    tgt_embedding, tgt_member = [], []
    with TrainingModeManager([Net.feature_extractor, Net.bottle_neck, Net.classifier],
                             train=False) as mgr, torch.no_grad():
        for i, (im_target, label_target) in enumerate(target_test_dl):
            im_target = im_target.to(output_device)
            _, feature_target, _ = Net(im_target)
            tgt_embedding.append(feature_target.detach().cpu().numpy())
            tgt_member.append(label_target.detach().cpu().numpy())
        tgt_embedding = np.concatenate(tgt_embedding, axis=0)
        tgt_member = np.concatenate(tgt_member, axis=0)
    return tgt_embedding, tgt_member


def merge_cluster(Cluster, tgt_embedding, tgt_member, tgt_predict_src, num_src_cls=20):
    tgt_embedding_k = tgt_embedding[tgt_predict_src < num_src_cls]
    tgt_embedding_unk = tgt_embedding[tgt_predict_src >= num_src_cls]
    tgt_predict_k = tgt_predict_src[tgt_predict_src < num_src_cls]
    unique_tgt_predict_k = np.unique(tgt_predict_k)
    tgt_predict = np.copy(tgt_predict_src)
    embedding = np.concatenate([tgt_embedding_k, tgt_embedding_unk], axis=0)
    tgt_member_new, _, _ = Cluster.fit_merge(embedding, tgt_predict_k, v1=False)
    mask = ~np.isin(tgt_member_new, unique_tgt_predict_k)
    tgt_member_new[mask] += num_src_cls
    tgt_predict[tgt_predict_src >= num_src_cls] = tgt_member_new
    return tgt_predict


def ExpWeight(step, gamma=3, max_iter=5000):
    step = max_iter-step
    ans = 1.0 * (np.exp(- gamma * step * 1.0 / max_iter))
    return float(ans)


class sim_bmm(object):
    def __init__(self, norm=False):
        self.bmm_model = BetaMixture1D()
        self.norm = norm
    
    def compute_probabilities_batch(self, sim_t, unk=1):
        sim_t[sim_t >= 1 - 1e-4] = 1 - 1e-4
        sim_t[sim_t <=  1e-4] = 1e-4
        B = self.bmm_model.posterior(sim_t, unk)
        return B

    def bmm_fit(self, sim_array):
        if self.norm:
            self.min = np.min(sim_array)
            self.max = np.max(sim_array)
            sim_array = (sim_array - self.min) / (self.max - self.min)

        sim_array[sim_array >= 1] = 1 - 10e-4
        sim_array[sim_array <= 0] = 10e-4
        self.bmm_model.fit(sim_array)
        self.bmm_model.create_lookup(1)

    def get_posterior(self, sim_t):
        '''
        out_t_free: detached tensor
        '''
        if self.norm:
            sim_t = (sim_t - self.min) / (self.max - self.min)
        w_unk_posterior = self.compute_probabilities_batch(sim_t, 1)
        w_k_posterior = 1 - w_unk_posterior
        return w_k_posterior, w_unk_posterior


