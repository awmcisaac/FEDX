# -*- coding: utf-8 -*-
# @Time    : 2022/9/17 21:14
# @Author  : LIU YI
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn.functional import softmax, cosine_similarity
import torch
from sklearn.decomposition import PCA
from kmeans_pytorch import kmeans
from sklearn.metrics import silhouette_score
v_dis = []
select = 256
max_round = 50
v_0_pre = None
v_1_pre = None
torch.random.manual_seed(2)
np.random.seed(2)

def calculate_v(data):
    u, s, v = torch.svd(data)
    sigma = torch.diag_embed(s)
    return u, sigma, v

    # #
    # r = torch.randperm(len(data))
    # data = data[r]
    # # data = data.t()
    # q, r = torch.linalg.qr(data)
    # # q = q[:,:select]
    # # r = r[:select]
    # # r_ = torch.flatten(r)
    # # r_ = r_.unsqueeze(0)
    # return q, r, r

def recreate_feature(data, b):
    A = torch.matmul(b, data.t())
    u, s, v = torch.svd(A)
    u = torch.matmul(v, u.t())
    # u = u[:, :select]
    # b = b[:select]
    # f = torch.matmul(u, b)
    return u

def dis_base(u1, b1, b2, data, select):
    u2 = recreate_feature(data, b2)
    data1 = torch.matmul(u1[:, :select], b1[:select])
    data2 = torch.matmul(u2[:, :select], b2[:select])
    return torch.dist(data1, data2).item()

def draw_evolution(index, data, name):
    record = []
    for i in range(len(data)):
        if i == 0:
            tep = data[i]
        else:
            # sig = data[i] < 0
            # sig_pre = tep < 0
            # sig = sig ^ sig_pre
            # if torch.sum(sig) > 0.5* torch.numel(sig):
            #     dis1 = torch.dist(-data[i], tep).item()
            # else:
            #     dis1 = torch.dist(data[i], tep).item()
            dis1 = torch.dist(data[i], tep).item()
            record.append(dis1)
            tep = data[i]
    index = index[0].cpu().tolist()
    del index[0]
    plt.plot(index, record)
    plt.title(name)
    plt.show()

def unit_symbol(data2):
    # data1_numpy = data1.cpu().numpy()
    # data2_numpy = data2.cpu().numpy()
    symble = []

    for i in range(len(data2)):
        # sig1 = data1[i] < 0
        sig2 = data2[i] > 0
        sig2 = (sig2.type(torch.int)-0.5)*2
        data2[i] = data2[i]*sig2
        symble.append(sig2)

    symble = torch.cat(symble, dim=0)
        # sig = sig1 ^ sig2
        # if torch.sum(sig) > 0.5*(torch.numel(sig)):
        #     data2[i] = -data2[i]

    return symble, data2

v1_collection = []
v2_collection = []
symbol1_collection = []
symbol2_collection = []

f1_distance = []
f2_distance = []

for round in range(max_round):
    # data1 = torch.load('./iid_ckpt/ckpt_aggregated/0_{}_0.pth'.format(str(round)))
    # data2 = torch.load('./iid_ckpt/ckpt_aggregated/1_{}_0.pth'.format(str(round)))
    data1 = torch.load('./ckpt_2_non_iid_ours/0_{}_0.pth'.format(str(round)))
    data2 = torch.load('./ckpt_2_non_iid_ours/1_{}_0.pth'.format(str(round)))

    # data = data.transpose(-2, -1)
    # data_ = data_.transpose(-2, -1)
    data1 = data1.detach()
    data2 = data2.detach()
    # pca = PCA(n_components=select)
    # pca.fit_transform(data)
    # data = pca.transform(data)
    u1, s1, v1 = calculate_v(data1)
    u2, s2, v2 = calculate_v(data2)
    b1 = torch.matmul(s1, v1.t())
    b2 = torch.matmul(s2, v2.t())

    dis = dis_base(u1, b1, b2, data1, select)
    f1_distance.append(dis)
    dis = dis_base(u2, b2, b1, data2, select)
    f2_distance.append(dis)
    #
    # b1_pre = b1
    # b2_pre = b2
plt.plot(f1_distance, label = 'client1')
plt.plot(f2_distance, label = 'client2')
plt.legend()
plt.show()
# print(f2_distance[-1])

    # symbol1, v1 = unit_symbol(v1)
    # symbol2, v2 = unit_symbol(v2)
    # v1_collection.append(v1.flatten().unsqueeze(0))
    # v2_collection.append(v2.flatten().unsqueeze(0))
    # symbol1_collection.append(symbol1.flatten().unsqueeze(0))
    # symbol2_collection.append(symbol2.flatten().unsqueeze(0))

# v1_collection = torch.cat(v1_collection, dim=0)
# v2_collection = torch.cat(v2_collection, dim=0)

# draw_evolution(v1_collection, 'client1')
# draw_evolution(v2_collection, 'client2')

# v1_collection_final = v1_collection[-40:]
# v2_collection_final = v2_collection[-40:]
#
# score1 = []
# score2 = []

# k_list = list(range(2,20))
#
# for k in k_list:
#     cluster_id1, cluster_center1 = kmeans(v1_collection_final, num_clusters=k, device='cuda')
#     cluster_id2, cluster_center2 = kmeans(v2_collection_final, num_clusters=k, device='cuda')
#     score1.append(silhouette_score(v1_collection_final.cpu(), cluster_id1))
#     score2.append(silhouette_score(v2_collection_final.cpu(), cluster_id2))
#
# plt.plot(k_list, score1, label = 'client1')
# plt.plot(k_list, score2, label = 'client2')
# # plt.xticks(k_list)
# plt.legend()
# plt.show()

# k = 4
# cluster_id1, cluster_center1 = kmeans(v1_collection, num_clusters=k, device='cuda')
# cluster_id2, cluster_center2 = kmeans(v2_collection, num_clusters=k, device='cuda')
# v1_collection_cluster = [v1_collection[cluster_id1 == i] for i in range(k)]
# v2_collection_cluster = [v2_collection[cluster_id2 == i] for i in range(k)]
#
# presenter1 = []
# presenter2 = []

# for one in range(len(v1_collection_cluster)):
#     # draw_evolution(torch.where(cluster_id1==one), v1_collection_cluster[one], 'client_1_{}'.format(one))
#     presenter1.append(v1_collection_cluster[one][-1])
# for one in range(len(v2_collection_cluster)):
#     # draw_evolution(torch.where(cluster_id2==one),v2_collection_cluster[one], 'client_2_{}'.format(one))
#     presenter2.append(v2_collection_cluster[one][-1])

# presenter_all = presenter1+presenter2
#
# distance = []
#
# for i in presenter_all:
#     tep = []
#     for j in presenter_all:
#         tep.append(torch.dist(i, j).item())
#     distance.append(tep)
#
# distance = np.array(distance)
# ax = plt.matshow(distance)
# plt.colorbar(ax.colorbar, fraction = 0.025)
#
# for (i, j), z in np.ndenumerate(distance):
#     plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
#
# plt.show()




# draw_evolution(v1_collection_a, 'client_1_cluster_a')
# draw_evolution(v1_collection_b, 'client_1_cluster_b')
# # draw_evolution(v1_collection, 'client_1')
#
# draw_evolution(v2_collection_a, 'client_2_cluster_a')
# draw_evolution(v2_collection_b, 'client_2_cluster_b')
#
# client1_a_presenter = v1_collection_a[-1]
# client1_b_presenter = v1_collection_b[-1]
# client2_a_presenter = v2_collection_a[-1]
# client2_b_presenter = v2_collection_b[-1]

# print(torch.dist(client1_a_presenter, ))

# print(torch.dist())

print('ok')
