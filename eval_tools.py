import numpy as np


def get_signal_match_cmc(gt_dist, gt_dist_is_inf, gallery_id, traj_id, max_rank=5):
    gallery_num = gallery_id.size
    traj_num = traj_id.size

    assert gt_dist.shape[0] == gallery_num and gt_dist.shape[1] == traj_num

    new_gt_dist = []
    new_gallery_id = []

    for g_i in range(gallery_num):
        matched_id = np.where(traj_id == gallery_id[g_i])
        assert len(matched_id) == 1
        matched_id = matched_id[0]
        # if matched_id.size == 1 and gt_dist_is_inf[g_i, matched_id]:  # true traj not exist
        #    continue
        if (gt_dist_is_inf[g_i] == True).all():
            #Visual trajectories that do not have the same timestamp with the wireless signal are excluded.
            continue
        if matched_id.size == 0:
            # Visual trajectories with no real wireless signals are eliminated.
            # No wireless signals are recorded for the pedestrians to whom the videos belonged.
            continue
        new_gt_dist.append(gt_dist[g_i])
        new_gallery_id.append(gallery_id[g_i])

    gt_dist = np.asarray(new_gt_dist)
    gallery_id = np.asarray(new_gallery_id)

    # ----

    gallery_num = gallery_id.size
    traj_num = traj_id.size
    all_cmc = []

    all_num = 0

    gt_dist = gt_dist.copy()

    assert gt_dist.shape[0] == gallery_num and gt_dist.shape[1] == traj_num
    if traj_num < max_rank:
        max_rank = traj_num

    indices = np.argsort(gt_dist, axis=1)
    gt_dist_sorted = np.sort(gt_dist, axis=1)
    matches = (traj_id[indices] == gallery_id[:, np.newaxis]).astype(np.int32)

    for g_i in range(gallery_num):
        cmc = matches[g_i].copy()
        assert not np.isinf(gt_dist_sorted[g_i, 0])
        cmc = cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        all_num += 1

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / all_num
    return all_cmc