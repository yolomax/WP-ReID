from video_reid_performance import compute_video_cmc_map
from update_module import get_vision_record_dist, visual_affinity_update, trajectory_distance_update
from eval_tools import get_match_cmc, norm_data
from copy import deepcopy
import numpy as np


def my_compute_video_cmc_map(*args):
    cmc, mAP = compute_video_cmc_map(*args, max_rank=20)
    cmc = np.round(cmc * 100, 2)
    mAP = np.round(mAP * 100, 2)
    return [cmc[0], cmc[4], cmc[9], cmc[19]], mAP


def my_get_match_cmc(*args, **kwargs):
    cmc = get_match_cmc(*args, **kwargs, max_rank=20)
    cmc = np.round(cmc * 100, 2)
    return [cmc[0], cmc[4], cmc[9], cmc[19]]


class RecurrentContextPropagationModule(object):
    def __init__(self, iters=4):
        self.iters = iters

    def unit(self):
        distMat_2 = visual_affinity_update(distMat, gt_dist=gt_gps_dist_1.copy(), T=t1, alpha=a)
        cmc, mAP = my_compute_video_cmc_map(distMat_2[query_index], query_info[:, 0], gallery_info[:, 0],
                                            query_info[:, 1], gallery_info[:, 1])
        print('2 GPS help ReID', mAP, cmc)

        plt_info[1, k_i, t1_i, t2_i, 0] = mAP
        plt_info[1, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)

        gt_gps_dist_2 = reid_help_gps(distMat_2, gt_gps_dist.copy(), k=k)
        cmc = my_get_match_cmc(gt_gps_dist_2[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                               query_id.copy(), traj_id.copy(),
                               vision_GPS_T=t2)

    def __call__(self, *args, **kwargs):




def update_with_gps(distMat, all_info, query_index, gallery_index, bndbox_gps, trajectory):
    '''
    :param distMat: (g, g), the distance between gallery tracklet with gallery tracklet
            :param all_info: (num, 5), num is the total number of gallery tracklet, each row [p_id, cam_id, idx, idx+frame_num, frame_num]
            :param query_index: the query index for query tracklet in gallery tracklet
            :param gallery_index: the gallery index for query tracklet in gallery tracklet
            :param bndbox_gps: (image_num, 3), each raw: [time_id, gps_j, gps_w]
            :param trajectory: dict, the key is the corresponding person id. trajectory[p_id] : (n, 3), each row [time_id, gps_j, gps_w]
            :return:
    '''
    distMat = norm_data(distMat)

    cmc, mAP = my_compute_video_cmc_map(distMat[query_index], all_info[query_index, 0], all_info[gallery_index, 0],
                                        all_info[query_index, 1], all_info[gallery_index, 1])

    print('Raw ReID Performance ', mAP, cmc)
    k_rang = list(range(7, 10))
    t1_rang = np.arange(74, 75, 1)
    t2_rang = np.arange(16, 17, 1)

    plt_info = np.zeros((8, len(k_rang), t1_rang.shape[0], t2_rang.shape[0], 9))

    traj_num = len(list(trajectory.keys()))

    query_info = all_info[query_index]
    query_id = query_info[:, 0]
    gallery_info = all_info[gallery_index]
    gallery_id = gallery_info[:, 0]

    gt_gps_dist, traj_id = get_vision_record_dist(gallery_info.copy(), bndbox_gps.copy(), deepcopy(trajectory))
    gt_dist_is_inf = np.isinf(gt_gps_dist)

    cmc = my_get_match_cmc(gt_gps_dist[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                           query_id.copy(), traj_id.copy(),
                           vision_GPS_T=16)
    print('Raw GT Performance ', cmc)

    for k_i, k in enumerate(k_rang):
        for a in np.arange(0.5, 0.6, 0.1):
            for t1_i, t1 in enumerate(t1_rang):
                for t2_i, t2 in enumerate(t2_rang):
                    distMat_1 = gps_help_reid(distMat.copy(), gt_dist=gt_gps_dist.copy(), T=t1, alpha=a)
                    cmc, mAP = my_compute_video_cmc_map(distMat_1[query_index], query_id, gallery_id,
                                                        query_info[:, 1], gallery_info[:, 1])
                    print(np.round(k, 2), np.round(a, 2), np.round(t1, 2), np.round(t2, 2))
                    print('1 GPS help ReID ', mAP, cmc)

                    plt_info[0, k_i, t1_i, t2_i, 0] = mAP
                    plt_info[0, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)

                    gt_gps_dist_1 = reid_help_gps(distMat_1, gt_gps_dist.copy(), k=k)
                    cmc = my_get_match_cmc(gt_gps_dist_1[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                                           query_id.copy(), traj_id.copy(), vision_GPS_T=t2)
                    print('1 ReID help GT match cmc ', cmc)
                    plt_info[0, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)

                    # 2

                    distMat_2 = gps_help_reid(distMat, gt_dist=gt_gps_dist_1.copy(), T=t1, alpha=a)
                    cmc, mAP = my_compute_video_cmc_map(distMat_2[query_index], query_info[:, 0], gallery_info[:, 0],
                                                        query_info[:, 1], gallery_info[:, 1])
                    print('2 GPS help ReID', mAP, cmc)

                    plt_info[1, k_i, t1_i, t2_i, 0] = mAP
                    plt_info[1, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)

                    gt_gps_dist_2 = reid_help_gps(distMat_2, gt_gps_dist.copy(), k=k)
                    cmc = my_get_match_cmc(gt_gps_dist_2[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                                           query_id.copy(), traj_id.copy(),
                                           vision_GPS_T=t2)
                    print('2 ReID help GT match cmc ', cmc)

                    plt_info[1, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)

                    # 3

                    distMat_3 = gps_help_reid(distMat, gt_dist=gt_gps_dist_2.copy(), T=t1, alpha=a)
                    cmc, mAP = my_compute_video_cmc_map(distMat_3[query_index], query_info[:, 0], gallery_info[:, 0],
                                                        query_info[:, 1], gallery_info[:, 1])
                    print('3 GPS help ReID ', mAP, cmc)

                    plt_info[2, k_i, t1_i, t2_i, 0] = mAP
                    plt_info[2, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)

                    gt_gps_dist_3 = reid_help_gps(distMat_3, gt_gps_dist.copy(), k=k)
                    cmc = my_get_match_cmc(gt_gps_dist_3[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                                           query_id.copy(), traj_id.copy(),
                                           vision_GPS_T=t2)
                    print('3 ReID help GT match cmc ', cmc)

                    plt_info[2, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)

                    # 4

                    distMat_4 = gps_help_reid(distMat, gt_dist=gt_gps_dist_3.copy(), T=t1, alpha=a)
                    cmc, mAP = my_compute_video_cmc_map(distMat_4[query_index], query_info[:, 0], gallery_info[:, 0],
                                                        query_info[:, 1], gallery_info[:, 1])
                    print('4 GPS help ReID ', mAP, cmc)

                    plt_info[3, k_i, t1_i, t2_i, 0] = mAP
                    plt_info[3, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)

                    gt_gps_dist_4 = reid_help_gps(distMat_4, gt_gps_dist.copy(), k=k)
                    cmc = my_get_match_cmc(gt_gps_dist_4[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                                           query_id.copy(), traj_id.copy(),
                                           vision_GPS_T=t2)
                    print('4 ReID help GT match cmc ', cmc)

                    plt_info[3, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)

                    # 5
                    #
                    # distMat_5 = gps_help_reid(distMat, gt_dist=gt_gps_dist_4.copy(), T=t1, alpha=a)
                    # cmc, mAP = my_compute_video_cmc_map(distMat_5[query_index], query_info[:, 0], gallery_info[:, 0],
                    #                                     query_info[:, 1], gallery_info[:, 1])
                    # print('5 GPS help ReID ', mAP, cmc)
                    #
                    # plt_info[4, k_i, t1_i, t2_i, 0] = mAP
                    # plt_info[4, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)
                    #
                    # gt_gps_dist_5 = reid_help_gps(distMat_5, gt_gps_dist.copy(), k=k)
                    # cmc = my_get_match_cmc(gt_gps_dist_5[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                    #                        query_id.copy(), traj_id.copy(),
                    #                        vision_GPS_T=t2)
                    # print('5 ReID help GT match cmc ', cmc)
                    #
                    # plt_info[4, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)
                    #
                    # # 6
                    #
                    # distMat_6 = gps_help_reid(distMat, gt_dist=gt_gps_dist_5.copy(), T=t1, alpha=a)
                    # cmc, mAP = my_compute_video_cmc_map(distMat_6[query_index], query_info[:, 0], gallery_info[:, 0],
                    #                                     query_info[:, 1], gallery_info[:, 1])
                    # print('6 GPS help ReID ', mAP, cmc)
                    #
                    # plt_info[5, k_i, t1_i, t2_i, 0] = mAP
                    # plt_info[5, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)
                    #
                    # gt_gps_dist_6 = reid_help_gps(distMat_6, gt_gps_dist.copy(), k=k)
                    # cmc = my_get_match_cmc(gt_gps_dist_6[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                    #                        query_id.copy(), traj_id.copy(),
                    #                        vision_GPS_T=t2)
                    # print('6 ReID help GT match cmc ', cmc)
                    #
                    # plt_info[5, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)
                    #
                    # # 7
                    #
                    # distMat_7 = gps_help_reid(distMat, gt_dist=gt_gps_dist_6.copy(), T=t1, alpha=a)
                    # cmc, mAP = my_compute_video_cmc_map(distMat_7[query_index], query_info[:, 0], gallery_info[:, 0],
                    #                                     query_info[:, 1], gallery_info[:, 1])
                    # print('7 GPS help ReID ', mAP, cmc)
                    #
                    # plt_info[6, k_i, t1_i, t2_i, 0] = mAP
                    # plt_info[6, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)
                    #
                    # gt_gps_dist_7 = reid_help_gps(distMat_7, gt_gps_dist.copy(), k=k)
                    # cmc = my_get_match_cmc(gt_gps_dist_7[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                    #                        query_id.copy(), traj_id.copy(),
                    #                        vision_GPS_T=t2)
                    # print('7 ReID help GT match cmc ', cmc)
                    #
                    # plt_info[6, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)
                    #
                    # # 8
                    #
                    # distMat_8 = gps_help_reid(distMat, gt_dist=gt_gps_dist_7.copy(), T=t1, alpha=a)
                    # cmc, mAP = my_compute_video_cmc_map(distMat_8[query_index], query_info[:, 0], gallery_info[:, 0],
                    #                                     query_info[:, 1], gallery_info[:, 1])
                    # print('8 GPS help ReID ', mAP, cmc)
                    #
                    # plt_info[7, k_i, t1_i, t2_i, 0] = mAP
                    # plt_info[7, k_i, t1_i, t2_i, 1:5] = np.asarray(cmc)
                    #
                    # gt_gps_dist_8 = reid_help_gps(distMat_8, gt_gps_dist.copy(), k=k)
                    # cmc = my_get_match_cmc(gt_gps_dist_8[query_index].copy(), gt_dist_is_inf[query_index].copy(),
                    #                        query_id.copy(), traj_id.copy(),
                    #                        vision_GPS_T=t2)
                    # print('8 ReID help GT match cmc ', cmc)
                    #
                    # plt_info[7, k_i, t1_i, t2_i, 5:9] = np.asarray(cmc)

    return plt_info