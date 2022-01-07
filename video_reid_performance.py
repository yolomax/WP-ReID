'''
The evaluation code is the python implementation of the evaluation code from https://github.com/liangzheng06/MARS-evaluation
'''


import numpy as np


def compute_AP(good_index, junk_index, order):
    """
    input:
    return:
    """
    cmc = np.zeros(order.size, dtype=np.float32)
    nGood = good_index.size
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0.0
    j = 0
    good_now = 0
    nJunk = 0
    for i_order, order_i in enumerate(order):
        flag = False
        if good_index[good_index == order_i].size != 0:
            cmc[i_order - nJunk:] = 1
            flag = True
            good_now += 1
        if junk_index[junk_index == order_i].size != 0:
            nJunk += 1
            continue

        if flag:
            intersect_size += 1.0
        recall = intersect_size / nGood
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j += 1
        if good_now == nGood:
            break
    return ap, cmc


def compute_video_cmc_map(distMat, q_pids, g_pids, q_camids, g_camids, junk_idx=None, max_rank=5):
    probe_num = q_pids.size
    gallery_num = g_pids.size
    max_rank = max_rank if gallery_num >= max_rank else gallery_num
    assert distMat.shape[0] == probe_num and distMat.shape[1] == gallery_num
    assert q_camids.size == probe_num and g_camids.size == gallery_num

    ap = np.zeros(probe_num, np.float32)
    cmc = np.zeros((probe_num, gallery_num), np.float32)
    for i_p in range(probe_num):
        dist = distMat[i_p, ...]
        p_id = q_pids[i_p]
        cam_p = q_camids[i_p]
        pos = np.where(g_pids == p_id)[0]
        pos2 = np.where(g_camids[pos] != cam_p)
        good_index = pos[pos2]
        pos3 = np.where(g_camids[pos] == cam_p)
        temp_junk_index = pos[pos3]
        junk_index = temp_junk_index if junk_idx is None else np.concatenate(
            (junk_idx, temp_junk_index), axis=0)
        dist_order = np.argsort(dist)
        ap[i_p, ...], cmc[i_p, ...] = compute_AP(good_index, junk_index, dist_order)

    cmc = np.sum(cmc, axis=0) / probe_num
    mAP = np.sum(ap) / ap.size

    return cmc[:max_rank], mAP