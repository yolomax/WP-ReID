import numpy as np
import copy
import logging
from prettytable import PrettyTable
from torch.utils.data import Dataset
from utils import np_filter
from wp_reid_dataset import WPReID


class DataBank(Dataset):
    def __init__(self, minframes):

        data_dict = WPReID().get_dict()

        self.images_dir_list = copy.deepcopy(data_dict['dir'])
        self.extra_data = data_dict['extra_data']
        self.test_only = data_dict['test_only']
        self.split_num = len(data_dict['split'])

        self.train_info, self.test_info, self.probe_index, self.gallery_index, self.junk_index = self._preprocess(data_dict, minframes)

        self.train_person_num = np.unique(self.train_info[:, 0]).size
        self.test_person_num = np.unique(self.test_info[:, 0]).size
        self.train_image_num = np.sum(self.train_info[:, 4])
        self.test_image_num = np.sum(self.test_info[:, 4])

        self.train_frames_len_min = np.min(self.train_info[:, 4])
        self.train_frames_len_max = np.max(self.train_info[:, 4])
        self.test_frames_len_min = np.min(self.test_info[:, 4])
        self.test_frames_len_max = np.max(self.test_info[:, 4])

        self.train_cam_num = np.unique(self.train_info[:, 1]).size
        self.test_cam_num = np.unique(self.test_info[:, 1]).size

    def _get_raw_data_info(self, data_dict):
        logger = logging.getLogger(__name__)
        logger.info(data_dict['info'])
        logger.info('Dataset creation time {}'.format(data_dict['time_tag']))
        data_dict = data_dict['split'][self.split_id]
        logger.info(data_dict['info'])
        train_info = np.asarray(data_dict['train'], np.int64)
        probe_info = np.asarray(data_dict['probe'], np.int64)
        gallery_info = np.asarray(data_dict['gallery'], np.int64)
        return train_info, probe_info, gallery_info

    def _check(self, train_info, test_info, probe_info, gallery_info):
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]
        logger = logging.getLogger(__name__)

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

            logger.info('After drop videos less than: train {:2d} test {:2d} frames, check cam number'.format(
                self.minframes['train'], self.minframes['test']))
            if len(probe_info_drop) > 0:
                for drop_info in probe_info_drop:
                    logger.warning('No related gallery track with different camera id. Drop probe ' + str(drop_info))
                probe_info = np.stack(probe_info_new)
                test_info = self._merge_to_test(probe_info, gallery_info)
            else:
                logger.info('All probe track have related gallery track with different camera id.')

        assert np.sum(train_info[:, 3] - train_info[:, 2] - train_info[:, 4]) == 0
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
                logger.warning('All gallery trackets have the same camera id with probe tracklet for ID： ' + str(p_id))

        assert np.unique(test_info[:, 2]).size == np.unique(np.concatenate((probe_info, gallery_info))[:, 2]).size
        if not self.test_only:
            assert np.intersect1d(train_info[:, 2], test_info[:, 2]).size == 0
        assert np.unique(train_info[:, 2]).size == train_info.shape[0]
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
        train_info, probe_info, gallery_info = self._get_raw_data_info(data_dict)
        test_info = self._merge_to_test(probe_info, gallery_info)

        if train_info[:, 4].max() > 1 or test_info[:, 4].max() > 1:
            self.is_image_dataset = False
            self.minframes = minframes
        else:
            self.is_image_dataset = True
            self.minframes = None

        self._print_info(train_info, test_info, probe_info, gallery_info, 'Raw')

        if self.minframes is not None:
            train_info = train_info[train_info[:, 4] >= self.minframes['train']]
            test_info = test_info[test_info[:, 4] >= self.minframes['test']]
            probe_info = probe_info[probe_info[:, 4] >= self.minframes['test']]
            gallery_info = gallery_info[gallery_info[:, 4] >= self.minframes['test']]

        test_info, probe_info = self._check(train_info, test_info, probe_info, gallery_info)

        if self.minframes is not None:
            self._print_info(train_info, test_info, probe_info, gallery_info, 'After Drop')

        probe_idx = DataBank._get_index(test_info, probe_info)
        gallery_idx = DataBank._get_index(test_info, gallery_info)

        junk_idx = np.where(test_info[:, 0] == -1)[0]
        train_info = self._get_pseudo_label(train_info)
        if not self.name == 'GPSReID':
            test_info = self._get_pseudo_label(test_info)
        return train_info, test_info, probe_idx, gallery_idx, junk_idx

    @staticmethod
    def _get_pseudo_label(track_info):
        pseudo_info = track_info.copy()
        real_pid = np.unique(track_info[:, 0])
        real_pid.sort()
        person_num = real_pid.size
        real_cid = np.unique(track_info[:, 1])
        real_cid.sort()
        cam_num = real_cid.size
        for pseudo_id in range(person_num):
            person_real_id = real_pid[pseudo_id]
            pseudo_info[track_info[:, 0] == person_real_id, 0] = pseudo_id
        for pseudo_id in range(cam_num):
            person_real_cid = real_cid[pseudo_id]
            pseudo_info[track_info[:, 1] == person_real_cid, 1] = pseudo_id
        return pseudo_info

    def __getitem__(self, item):
        info = {}
        for k in item.keys():
            if k == 'idx_list':
                data = self._storemanager.read(item['idx_list'])
                data = self.transform(data)
                info['data'] = data
            else:
                info[k] = item[k]
        return info

    def _print_info(self, train_info, test_info, probe_info, gallery_info, extra_info: str):

        GalleryInds = np.unique(gallery_info[:, 0])
        probeInds = np.unique(probe_info[:, 0])
        logger = logging.getLogger(__name__)
        logger.info('Gallery ID diff Probe ID: %s' % np.setdiff1d(GalleryInds, probeInds))

        table = PrettyTable([extra_info, 'Train', 'Test', 'Probe', 'Gallery'])

        table.add_row(['#ID', np.unique(train_info[:, 0]).size,
                       np.unique(test_info[:, 0]).size,
                       np.unique(probe_info[:, 0]).size,
                       np.unique(gallery_info[:, 0]).size])
        table.add_row(['#Track', train_info.shape[0],
                       test_info.shape[0],
                       probe_info.shape[0],
                       gallery_info.shape[0]])
        table.add_row(['#Image', np.sum(train_info[:, 4]),
                       np.sum(test_info[:, 4]),
                       np.sum(probe_info[:, 4]),
                       np.sum(gallery_info[:, 4])])
        table.add_row(['#Cam', np.unique(train_info[:, 1]).size,
                       np.unique(test_info[:, 1]).size,
                       np.unique(probe_info[:, 1]).size,
                       np.unique(gallery_info[:, 1]).size])
        table.add_row(['MaxLen', np.max(train_info[:, 4]),
                       np.max(test_info[:, 4]),
                       np.max(probe_info[:, 4]),
                       np.max(gallery_info[:, 4])])
        table.add_row(['MinLen', np.min(train_info[:, 4]),
                       np.min(test_info[:, 4]),
                       np.min(probe_info[:, 4]),
                       np.min(gallery_info[:, 4])])
        table.add_row(['AvgLen', int(np.mean(train_info[:, 4])),
                       int(np.mean(test_info[:, 4])),
                       int(np.mean(probe_info[:, 4])),
                       int(np.mean(gallery_info[:, 4]))])

        logger.info('\n%s' % table)

    def __len__(self):
        logger = logging.getLogger(__name__)
        logger.warning('-------The length of dataset is no meaning!---------')
        return self.train_info.shape[0] + self.test_info.shape[0]