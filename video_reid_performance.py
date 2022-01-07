'''
The evaluation code is the python implementation of the evaluation code from https://github.com/liangzheng06/MARS-evaluation
'''


import numpy as np


def compute_AP(good_index, junk_index, order):
    """
    Compute Average Precision(AP) and CMC for one query.\\
    AP is defined as the area under the Precision-Recall(PR) curve.

    **params**:
    good_index: 1-d array; indexs that record all good prediction;\\
    junk_index: 1-d array; indexs that record all junk predictions:\\
    order: 1-d array; ordered indexs; 

    **return**:
    ap: a scalar;\\
    cmc: 1-d array; size(cmc) = (order,);

    **example**: \\
    a list with good/junk prediction. (1: good; 0: junk) \\
    order = [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]\\
    good_index == [0, 1, 3, 4, 7,];\\
    junk_index == [2, 5, 6, 8, 9, 10, 11,];\\
    >> cmc == [] \\
    >> ap == 
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
    junk_idx: \\
    mark_rank: 

    return:

    cmc: array_like, size: N; Cumulative Match Characteristics \\
    map: scalar; mean Average Precision 
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
        p_id = q_pids[i_p]  # probe person id
        cam_p = q_camids[i_p]   # 
        pos = np.where(g_pids == p_id)[0]   # np.where(condition) returns condition satisfies.
        pos2 = np.where(g_camids[pos] != cam_p)
        good_index = pos[pos2]
        pos3 = np.where(g_camids[pos] == cam_p)
        temp_junk_index = pos[pos3]
        junk_index = temp_junk_index if junk_idx is None else np.concatenate(
            (junk_idx, temp_junk_index), axis=0)
        dist_order = np.argsort(dist)
        ap[i_p, ...], cmc[i_p, ...] = compute_AP(good_index, junk_index, dist_order) # recall "compute_AP" function

    cmc = np.sum(cmc, axis=0) / probe_num
    mAP = np.sum(ap) / ap.size

    return cmc[:max_rank], mAP