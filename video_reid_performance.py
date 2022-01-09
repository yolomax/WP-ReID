'''
The evaluation code is the python implementation of the evaluation code from https://github.com/liangzheng06/MARS-evaluation
'''

import numpy as np


def compute_AP(good_index, junk_index, order):
    """
    Compute Average Precision(AP) and CMC for one query.\\
    AP is defined as the area under the Precision-Recall(PR) curve.

    params:
    good_index: 1-d array; indexs that record all good images;\\
    junk_index: 1-d array; indexs that record all junk images:\\
    order: 1-d array; index record the distance from small to large; 

    return:
    ap: a scalar;\\
    cmc: 1-d array;
    """
    cmc = np.zeros(order.size, dtype=np.float32)
    nGood = good_index.size # total number of good prediction
    old_recall = 0.0    # init recall, then recall increase util 1
    old_precision = 1.0 # init precision, then precision decrease
    ap = 0.0    # init ap
    intersect_size = 0.0
    j = 0
    good_now = 0    # the number of good image (good prediction)
    nJunk = 0       # the number of junk image
    for i_order , order_i in enumerate(order):
        flag = False    # good/junk index, default "junk"
        if good_index[good_index == order_i].size != 0:
            cmc[i_order - nJunk:] = 1   # after a good index, cmc == 1
            flag = True
            good_now += 1
        if junk_index[junk_index == order_i].size != 0:
            nJunk += 1
            continue

        if flag:
            intersect_size += 1.0
        recall = intersect_size / nGood # TP / TP + TN
        precision = intersect_size / (j + 1) # TP / TP + FP
        ap += (recall - old_recall) * ((old_precision + precision) / 2) # area under PR curve
        old_recall = recall # update
        old_precision = precision # update
        j += 1
        if good_now == nGood:   # all good image is recalled.
            break
    return ap, cmc


def compute_video_cmc_map(distMat, q_pids, g_pids, q_camids, g_camids, junk_idx=None, max_rank=5):
    """
    params:

    distMat: 2-d array_like, size: M x N (M: number of probe, N: Number of gallary)\\
    q_pids: array_like, size: M; query person ID\\
    g_pids: array_like, size: N; gallary person ID \\
    q_camids: array_like, size: M; query camera ID \\
    g_camids: array_like, size: N; gallary camera ID \\
    junk_idx: index of junk image. \\
    max_rank: the maximum of rank numbers. \\

    return:

    cmc: array_like, size: N; Cumulative Match Characteristics \\
    map: scalar; mean Average Precision 

    Q: What is good / junk image?
    A: Good image is defined as the gallary images with the same ID with the probe, but different camera ID;
    while junk image gets the same ID with the probe. 

    """
    probe_num = q_pids.size
    gallery_num = g_pids.size
    max_rank = max_rank if gallery_num >= max_rank else gallery_num  
    assert distMat.shape[0] == probe_num and distMat.shape[1] == gallery_num
    assert q_camids.size == probe_num and g_camids.size == gallery_num

    ap = np.zeros(probe_num, np.float32)    # multiple query AP; size M
    cmc = np.zeros((probe_num, gallery_num), np.float32)    # multiple query CMC; size: M x N
    for i_p in range(probe_num):    # the i_p ^th probe image
        dist = distMat[i_p, ...]
        p_id = q_pids[i_p]  # one person id
        cam_p = q_camids[i_p]   # one camera id 
        pos = np.where(g_pids == p_id)[0]   # find the index where Person ID is the same as probe. (U should know how np.where works)
        pos2 = np.where(g_camids[pos] != cam_p) # find good image
        good_index = pos[pos2]  # tuple as index
        pos3 = np.where(g_camids[pos] == cam_p) # find junk image
        temp_junk_index = pos[pos3]
        junk_index = temp_junk_index if junk_idx is None else np.concatenate(
            (junk_idx, temp_junk_index), axis=0)
        dist_order = np.argsort(dist) # return index 
        ap[i_p, ...], cmc[i_p, ...] = compute_AP(good_index, junk_index, dist_order) # recall "compute_AP" function

    cmc = np.sum(cmc, axis=0) / probe_num
    mAP = np.sum(ap) / ap.size

    return cmc[:max_rank], mAP