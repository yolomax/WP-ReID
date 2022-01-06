import numpy as np


def norm_data(a):
    a = a.copy()
    a_min = np.min(a, axis=1, keepdims=True)
    _range = np.max(a,axis=1,keepdims=True) - a_min
    return (a - a_min) / _range


def trans_gps_diff_to_dist(a, b):
    dist = a - b
    j_dist = dist[0] * 111000 * np.cos(a[1] / 180 * np.pi)
    w_dist = dist[1] * 111000
    return np.sqrt(np.power(j_dist, 2)+np.power(w_dist, 2))


def find_bndbox_gps(s, box_info):
    output = []
    for track_i in s:
        idx_list = np.arange(track_i[2], track_i[3])
        i_gps_info = box_info[idx_list]
        output.append(i_gps_info)
    assert len(output) == s.shape[0]
    return output


def get_vision_record_dist(gallery_info, bndbox_gps, trajectory):
    gallery_box_gps = find_bndbox_gps(gallery_info, bndbox_gps)
    traj_id = list(trajectory.keys())
    dist = np.zeros((len(gallery_box_gps), len(traj_id)), dtype=np.float64)
    for i_p, p_i in enumerate(traj_id):
        p_data = trajectory[p_i].copy()
        p_time_id = p_data[:, 0].copy().astype(np.int64)
        for i_track, track_i in enumerate(gallery_box_gps):
            dist_tmp = []
            for box_i in track_i:
                time_id = int(box_i[0])
                match_idx = np.argwhere(p_time_id == time_id)
                if match_idx.size > 0:
                    assert match_idx.size == 1
                    match_idx = match_idx[0, 0]
                    match_data = p_data[match_idx]
                    dist_tmp.append(trans_gps_diff_to_dist(box_i[1:3], match_data[1:3]))
            if len(dist_tmp) > 0:
                dist[i_track, i_p] = np.asarray(dist_tmp).mean()
            else:
                dist[i_track, i_p] = np.inf
    return dist, np.asarray(traj_id, np.int64)


def trajectory_distance_update(distMat, gt_gps_dist, k=5):

    assert gt_gps_dist.shape[0] == distMat.shape[0]
    
    gt_dist_sorted = np.sort(gt_gps_dist, axis=1)   
    #sorted: to get the smallest distance between one video sequence and a wireless traj.
    # if this value is inf, then this video sequence doesn't get a corresponding traj.
    idx_selected = []
    for g_i in range(gt_gps_dist.shape[0]):    # remove the video having no traj
        if np.isinf(gt_dist_sorted[g_i, 0]):
            continue
        idx_selected.append(g_i)
    idx_selected = np.asarray(idx_selected)

    distMat = distMat[idx_selected][:, idx_selected].copy() # why not distMat[idx_selected][idx_selected]?
    gt_gps_dist_raw = gt_gps_dist.copy()    # return value, we only need to refine part of this matrix.
    gt_gps_dist = gt_gps_dist[idx_selected].copy()

    #raw_gt_dist = gt_gps_dist.copy()
    idx = np.argsort(distMat, axis=1)[:, :k]    # help np.argsort(x): return the index of sorted value (each row);
    #the set corresponds to \phi K-nearest set
    distMat_sorted = np.sort(distMat, axis=1)[:, :k]
    tg_dist = gt_gps_dist.T # matrix transpose
    query_num = distMat.shape[0]

    gt_dist_wighted = []
    for q_i in range(query_num):    # q_i the column index, representing the distance between i^th video sequence and others.
        tg_dist_i = tg_dist[:, idx[q_i]].copy()     
        dist_weight = 1 - distMat_sorted[q_i]   # find weight(vector) for every q_i

        dist_weight = np.expand_dims(dist_weight, axis=0)
        dist_weight = np.repeat(dist_weight, tg_dist_i.shape[0], axis=0)
        dist_weight[tg_dist_i == np.inf] = 0
        #tg_dist_i[tg_dist_i == np.inf] = 0
        dist_weight_sum = dist_weight.sum(axis=1, keepdims=True)

        ident_idx = dist_weight_sum == 0
        ident_idx = ident_idx.reshape(-1)

        dist_weight = dist_weight / (dist_weight_sum + 1e-12)   # normalize the weight vector

        tg_dist_wighted_i = tg_dist_i * dist_weight
        tg_dist_wighted_i = tg_dist_wighted_i.sum(axis=1)

        tg_dist_wighted_i[ident_idx] = tg_dist[:, q_i][ident_idx]

        gt_dist_wighted.append(tg_dist_wighted_i)
    gt_dist_wighted = np.asarray(gt_dist_wighted)

    gt_gps_dist_raw[idx_selected] = gt_dist_wighted

    return gt_gps_dist_raw


def visual_affinity_update(distMat, gt_dist, T, alpha=0.5):
    #distMat corresponds to normalized F_{i, j} ?
    #gt_dis are the matrix that record the distance between each video squence(trajectory) and each wireless trajectory.
    #gt_dis size: (N x M) 
    #gps_disMat are the matrix that record \hat D_{i, j}
    #gps_disMat size: (N x N)
    gps_distMat = np.zeros(distMat.shape, dtype=distMat.dtype)
    assert distMat.shape[0] == gt_dist.shape[0] and distMat.shape[1] == gt_dist.shape[0] # == N (the number of video sequences)
    #calcalate \hat D_{i, j} i, j are lower index of two video sequences
    #recall the definition of \hat D_{i, j}: ...
    for q in range(distMat.shape[0]):
        gt_dist_q = gt_dist[q]
        for g in range(distMat.shape[1]):
            gt_dist_g = gt_dist[g]
            avg_dist = (gt_dist_q + gt_dist_g) / 2.0
            gps_distMat[q, g] = avg_dist.min()

    idx = gps_distMat <= T #threshold
    gps_distMat[idx] /= T
    S = 1 - distMat
    S = S.copy()
    #update cases gps_dist <= threshold
    S[idx] = S[idx] * (1 - alpha) + (1 - gps_distMat[idx]) * alpha
    #do not update cases i == j
    for i in range(distMat.shape[0]):
        S[i,i] = 1 - distMat[i, i]
    return 1 - S