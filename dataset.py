import numpy as np
import copy
import logging
from prettytable import PrettyTable
from utils import np_filter
from wp_reid_dataset import WPReID


class DataBank(object):
    def __init__(self, minframes):

        data_dict = WPReID().get_dict()

        self.images_dir_list = copy.deepcopy(data_dict['dir'])
        self.gps_info = data_dict['gps']
        self.trajectory_info = data_dict['trajectory']
        self.test_info, self.probe_index, self.gallery_index, self.junk_index = self._preprocess(data_dict, minframes)

        self.test_person_num = np.unique(self.test_info[:, 0]).size
        self.test_image_num = np.sum(self.test_info[:, 4])

        self.test_frames_len_min = np.min(self.test_info[:, 4])
        self.test_frames_len_max = np.max(self.test_info[:, 4])
        self.test_cam_num = np.unique(self.test_info[:, 1]).size

    def _get_raw_data_info(self, data_dict):
        print(data_dict['info'])
        probe_info = np.asarray(data_dict['probe'], np.int64)
        gallery_info = np.asarray(data_dict['gallery'], np.int64)
        return probe_info, gallery_info

    def _check(self, test_info, probe_info, gallery_info):
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]

        if self.minframes is not None:
            probe_info_new = []
            probe_info_drop = []
            for probe_i in range(probe_info.shape[0]):
                data_info = probe_info[probe_i]
                p_id = data_info[0]
                p_cam_id = data_info[1]
                g_info = np_filter(gallery_info, [p_id])
                g_cam_id = np.unique(g_info[:, 1])
                if np.setdiff1d(g_cam_id, np.asarray([p_cam_id])).size == 0:  # there is no tracklet of this person in the gallery set with different camera id.
                    probe_info_drop.append(data_info)
                else:
                    probe_info_new.append(data_info)

            print('After drop videos less than: test {:2d} frames, check cam number'.format(self.minframes))
            if len(probe_info_drop) > 0:
                for drop_info in probe_info_drop:
                    print('No related gallery track with different camera id. Drop probe ' + str(drop_info))
                probe_info = np.stack(probe_info_new)
                test_info = self._merge_to_test(probe_info, gallery_info)
            else:
                print('All probe track have related gallery track with different camera id.')

        assert np.sum(test_info[:, 3] - test_info[:, 2] - test_info[:, 4]) == 0
        assert np.sum(probe_info[:, 3] - probe_info[:, 2] - probe_info[:, 4]) == 0
        assert np.sum(gallery_info[:, 3] - gallery_info[:, 2] - gallery_info[:, 4]) == 0

        test_id = np.unique(test_info[:, 0])
        probe_id = np.unique(probe_info[:, 0])
        gallery_id = np.unique(gallery_info[:, 0])
        assert -1 not in set(test_id)   # junk id set to be -1, it should have been removed.

        assert np.setdiff1d(probe_id, gallery_id).size == 0
        assert set(test_id) == set(probe_id).union(set(gallery_id))

        for probe_i in range(probe_info.shape[0]):
            data_info = probe_info[probe_i]
            p_id = data_info[0]
            p_cam_id = data_info[1]
            g_info = np_filter(gallery_info, [p_id])
            g_cam_id = np.unique(g_info[:, 1])
            if not np.setdiff1d(g_cam_id, np.asarray([p_cam_id])).size > 0:
                print('All gallery trackets have the same camera id with probe tracklet for ID： ' + str(p_id))

        assert np.unique(test_info[:, 2]).size == np.unique(np.concatenate((probe_info, gallery_info))[:, 2]).size
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]
        assert np.unique(probe_info[:, 2]).size == probe_info.shape[0]
        assert np.unique(gallery_info[:, 2]).size == gallery_info.shape[0]
        return test_info, probe_info

    @staticmethod
    def _get_index(rawset, subset):
        index = []
        for i_probe in range(subset.shape[0]):
            begin = subset[i_probe, 2]
            temp_index = np.where(rawset[:, 2] == begin)[0]
            assert temp_index.size == 1
            temp_index = temp_index[0]
            index.append(temp_index)
        index = np.asarray(index, dtype=np.int64)
        return index

    def _merge_to_test(self, probe_info, gallery_info):
        begin_idx_box = gallery_info[:, 2].tolist()
        temp_info = []
        for probe_i in range(probe_info.shape[0]):
            probe_i_info = probe_info[probe_i]
            if probe_i_info[2] not in begin_idx_box:
                temp_info.append(probe_i_info)
                begin_idx_box.append(probe_i_info[2])

        if len(temp_info) == 0:
            return gallery_info.copy()
        else:
            temp_info = np.asarray(temp_info, np.int64)
            test_info = np.concatenate((temp_info, gallery_info), axis=0)
            return test_info

    def _preprocess(self, data_dict, minframes):
        probe_info, gallery_info = self._get_raw_data_info(data_dict)
        test_info = self._merge_to_test(probe_info, gallery_info)

        if test_info[:, 4].max() > 1:
            self.is_image_dataset = False
            self.minframes = minframes
        else:
            self.is_image_dataset = True
            self.minframes = None

        self._print_info(test_info, probe_info, gallery_info, 'Raw')

        if self.minframes is not None:
            test_info = test_info[test_info[:, 4] >= self.minframes]
            probe_info = probe_info[probe_info[:, 4] >= self.minframes]
            gallery_info = gallery_info[gallery_info[:, 4] >= self.minframes]

        test_info, probe_info = self._check(test_info, probe_info, gallery_info)

        if self.minframes is not None:
            self._print_info(test_info, probe_info, gallery_info, 'After Drop')

        probe_idx = DataBank._get_index(test_info, probe_info)
        gallery_idx = DataBank._get_index(test_info, gallery_info)

        junk_idx = np.where(test_info[:, 0] == -1)[0]
        return test_info, probe_idx, gallery_idx, junk_idx

    def _print_info(self, test_info, probe_info, gallery_info, extra_info: str):

        GalleryInds = np.unique(gallery_info[:, 0])
        probeInds = np.unique(probe_info[:, 0])
        print('Gallery ID diff Probe ID: %s' % np.setdiff1d(GalleryInds, probeInds))

        table = PrettyTable([extra_info, 'Test', 'Probe', 'Gallery'])

        table.add_row(['#ID',
                       np.unique(test_info[:, 0]).size,
                       np.unique(probe_info[:, 0]).size,
                       np.unique(gallery_info[:, 0]).size])
        table.add_row(['#Track',
                       test_info.shape[0],
                       probe_info.shape[0],
                       gallery_info.shape[0]])
        table.add_row(['#Image',
                       np.sum(test_info[:, 4]),
                       np.sum(probe_info[:, 4]),
                       np.sum(gallery_info[:, 4])])
        table.add_row(['#Cam',
                       np.unique(test_info[:, 1]).size,
                       np.unique(probe_info[:, 1]).size,
                       np.unique(gallery_info[:, 1]).size])
        table.add_row(['MaxLen',
                       np.max(test_info[:, 4]),
                       np.max(probe_info[:, 4]),
                       np.max(gallery_info[:, 4])])
        table.add_row(['MinLen',
                       np.min(test_info[:, 4]),
                       np.min(probe_info[:, 4]),
                       np.min(gallery_info[:, 4])])
        table.add_row(['AvgLen',
                       int(np.mean(test_info[:, 4])),
                       int(np.mean(probe_info[:, 4])),
                       int(np.mean(gallery_info[:, 4]))])

        print('\n%s' % table)
