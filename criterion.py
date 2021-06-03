import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.task = cfg['task_type']
        self.criterion = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def forward(self, input_var):
        '''
        embeddings_kp: [num_keypoints, k] in one batch
        labels_kp: [num_keypoints]
        pcds: [B, N, 3]
        embeddings: [B, N, k]
        mean_distance: [B, 78, 78]
        evecs: [B]
        '''
        embeddings_kp, labels_kp, pcds, embeddings, mean_distance, evecs = input_var
        margin = float(self.cfg['margin']) # margin = 1.
        dist_thr = self.cfg['dist_thr'] # dist_thr = -1
        if dist_thr > 0:
            loss_fn = OnlineContrastiveLoss(
                margin, HardNegativePairSelectorAdaptive(mean_distance, dist_thr), None)
        else:
            loss_fn = OnlineContrastiveLoss(
                margin, HardNegativePairSelectorAdaptive(mean_distance, -1), mean_distance)

        loss_kp = loss_fn(embeddings_kp, labels_kp)
        loss_total = loss_kp
        loss = {}

        if self.cfg['laplacian_reg']:
            loss_fn = LaplacianLoss(0.5, 0.8)
            loss_lap = loss_fn(embeddings, evecs)
            loss_total += 0.05 * loss_lap
            loss['lap'] = loss_lap
        
        loss['total'] = loss_total
        return loss

class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class HardNegativePairSelectorAdaptive(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, mean_distance, dist_thr, cpu=True):
        super(HardNegativePairSelectorAdaptive, self).__init__()
        self.cpu = cpu
        self.mean_distance = mean_distance[0].cpu().numpy()
        self.dist_thr = dist_thr

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings) # the distance between each embedding

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(
            labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(
            labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        negative_distances = distance_matrix[negative_pairs[:,
                                                            0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()

        labels_1 = tuple(labels[negative_pairs[:, 0]].tolist())
        labels_2 = tuple(labels[negative_pairs[:, 1]].tolist())
        label_pair = (labels_1, labels_2)
        geo_distances = self.mean_distance[label_pair]

        if self.dist_thr > 0:
            # Ignore pairs with dist < self.dist_thr
            inds = np.where(geo_distances > self.dist_thr)[0].tolist()
            negative_distances = negative_distances[inds]
        else:
            negative_distances = - \
                np.maximum(geo_distances - negative_distances, 0.)

        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[
            :len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector, mean_distance=None):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        if mean_distance is not None:
            self.mean_distance = mean_distance[0].cuda()
        else:
            self.mean_distance = None

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] -
                         embeddings[positive_pairs[:, 1]]).pow(2).sum(1)

        labels_1 = tuple(target[negative_pairs[:, 0]].tolist())
        labels_2 = tuple(target[negative_pairs[:, 1]].tolist())
        label_pair = (labels_1, labels_2)

        if self.mean_distance is not None:
            # print(np.sort(self.mean_distance[label_pair].cpu().numpy()))
            # print('mean dist',self.mean_distance[label_pair].mean())
            negative_loss = F.relu(
                self.mean_distance[label_pair] - ((embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                    1) + 1e-6).sqrt()).pow(2)
        else:
            negative_loss = F.relu(
                self.margin - ((embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                    1) + 1e-6).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        # print('positive', positive_loss.mean())
        # print('negative', negative_loss.mean())
        return loss.mean()

class LaplacianLoss(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, embeddings, evecs):
        """
        embeddings: [b, num_points, dim_feat]
        """
        batch_size = embeddings.size(0)
        embeddings = embeddings - embeddings.mean(dim=-1).unsqueeze(-1)
        embeddings = embeddings / (torch.norm(embeddings, dim=-1).unsqueeze(-1).repeat(1,1, embeddings.size(-1)))
        embeddings[embeddings != embeddings] = 0.
        cov_mat = torch.matmul(embeddings.permute(0, 2, 1), embeddings)
        cov_mat_s = torch.matmul(cov_mat, cov_mat)
        trace = 0.
        for i in range(batch_size):
            trace += torch.trace(cov_mat_s[i])
        trace /= batch_size
        return 1. - trace / (embeddings.size(1) **2)

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix