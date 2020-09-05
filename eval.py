from video_reid_performance import compute_video_cmc_map
from update_module import get_vision_record_dist, visual_affinity_update, trajectory_distance_update, norm_data
from eval_tools import get_signal_match_cmc
from copy import deepcopy
import numpy as np


class RecurrentContextPropagationModule(object):
    def __init__(self, k, delta, iters, distMat, gt_dist, traj_id, alpha=0.5):
        self.iters = iters
        self.k = k
        self.delta = delta
        self.alpha = alpha

        self.traj_id = traj_id
        self.raw_distMat = norm_data(distMat)
        self.raw_gt_dist = gt_dist
        self.gt_dist_is_inf = np.isinf(self.raw_gt_dist)

    def init_dataset_info(self, all_info, query_index, gallery_index):
        self.query_index = query_index
        query_info = all_info[self.query_index]
        gallery_info = all_info[gallery_index]
        self.query_id = query_info[:, 0]
        self.gallery_id = gallery_info[:, 0]
        self.query_cam_id = query_info[:, 1]
        self.gallery_cam_id = gallery_info[:, 1]

    def rcpm(self, gt_dist_new, iteration=1):
        distMat_new = visual_affinity_update(self.raw_distMat, gt_dist=gt_dist_new.copy(), T=self.delta, alpha=self.alpha)
        cmc_reid, mAP_reid = compute_video_cmc_map(distMat_new[self.query_index], self.query_id, self.gallery_id,
                                            self.query_cam_id, self.gallery_cam_id)
        gt_dist_new = trajectory_distance_update(distMat_new, self.raw_gt_dist.copy(), k=self.k)
        cmc_SM = get_signal_match_cmc(gt_dist_new[self.query_index].copy(), self.gt_dist_is_inf[self.query_index].copy(),
                               self.query_id.copy(), self.traj_id.copy())
        print('Iteration {}: ReID rank-1 {:.2f} mAP {:.2f}. Signal Matching rank-1 {:.2f}'.format(iteration,
                                                                                                  cmc_reid[0] * 100,
                                                                                                  mAP_reid * 100,
                                                                                                  cmc_SM[0] * 100))
        return gt_dist_new

    def __call__(self, *args, **kwargs):
        print('K={}, Delta={}, Iteration={}'.format(self.k, self.delta, self.iters))
        cmc_reid, mAP_reid = compute_video_cmc_map(self.raw_distMat[self.query_index], self.query_id, self.gallery_id,
                                         self.query_cam_id, self.gallery_cam_id)
        cmc_SM = get_signal_match_cmc(self.raw_gt_dist[self.query_index].copy(), self.gt_dist_is_inf[self.query_index].copy(),
                               self.query_id.copy(), self.traj_id.copy())

        print('Iteration {}: ReID rank-1 {:.2f} mAP {:.2f}. Signal Matching rank-1 {:.2f}'.format(0, cmc_reid[0] * 100,
                                                                                                  mAP_reid * 100,
                                                                                                  cmc_SM[0] * 100))
        last_gt_dist = self.raw_gt_dist
        for i in range(self.iters):
            last_gt_dist = self.rcpm(last_gt_dist, iteration=i+1)


def update_with_gps(distMat, all_info, query_index, gallery_index, bndbox_gps, trajectory, k=9, delta=74, iters=5):
    '''
    :param distMat: (g, g), the distance between gallery tracklet with gallery tracklet
            :param all_info: (num, 5), num is the total number of gallery tracklet, each row [p_id, cam_id, idx, idx+frame_num, frame_num]
            :param query_index: the query index for query tracklet in gallery tracklet
            :param gallery_index: the gallery index for query tracklet in gallery tracklet
            :param bndbox_gps: (image_num, 3), each raw: [time_id, gps_j, gps_w]
            :param trajectory: dict, the key is the corresponding person id. trajectory[p_id] : (n, 3), each row [timestamp, gps_j, gps_w]
            :return:
    '''
    gallery_info = all_info[gallery_index]
    gt_gps_dist, traj_id = get_vision_record_dist(gallery_info.copy(), bndbox_gps.copy(), deepcopy(trajectory))
    rcpm = RecurrentContextPropagationModule(k=k, delta=delta, iters=iters, distMat=distMat, gt_dist=gt_gps_dist, traj_id=traj_id)
    rcpm.init_dataset_info(all_info=all_info, query_index=query_index, gallery_index=gallery_index)
    rcpm()

