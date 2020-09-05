from pathlib import Path
from utils import DataPacker, file_abs_path
import numpy as np


class WPReID(object):
    def __init__(self):
        self.raw_data_folder = Path('/data')
        self.cropped_images_dir = self.raw_data_folder / 'cropped_data'
        # Where you put the images data. You can contact us by email for this dataset.

        self.split_file_dir = file_abs_path(__file__) / 'files' / 'wp_reid_info.json'

        self.all_image_num = 106578
        self.resize_hw = (256, 128)

        self.check_raw_file()

    def check_raw_file(self):
        # assert self.cropped_images_dir.exists()
        assert self.split_file_dir.exists()

    def get_dict(self):

        info = DataPacker.load(self.split_file_dir)
        img_dir_list = info['img_dir']

        img_dir_list = [str(self.raw_data_folder / i) for i in img_dir_list]
        '''
                img_dir_list is a list of all the images in the dataset.
                Each image is a cropped person image. 
                '''

        probe_info = np.asarray(info['probe'], dtype=np.int64)
        '''
                shape [P, 5], P is the number of query videos.

                probe_info[p, :] is a record of a certain video of a pedestrian.
                probe_info[p, 0] is the person ID.
                probe_info[p, 1] is the camera ID.
                probe_info[p, 2] is the index of the starting frame of the video in the picture address list.
                probe_info[p, 3] is the index of the end frame of the video in the picture address list.
                probe_info[p, 4] is the total number of video frames.  probe_info[p, 4] = probe_info[p, 3] - probe_info[p, 2]

                All frames of a video are stored in the list in chronological order, 
                so when you want to get the address of all frames of video p, 
                you can refer to the following code:

                video_all_frames_P = [img_dir_list[i] for i in range(probe_info[p, 2], probe_info[p, 3])]

                '''
        gallery_info = np.asarray(info['gallery'], dtype=np.int64)
        '''
                shape [G, 5], G is the number of gallery videos.
                The definition of each of its items is the same as probe_info.
                If you want to get all the frame addresses for the gallery video g, you can refer to the code below

                video_all_frames_g = [img_dir_list[i] for i in range(gallery_info[g, 2], gallery_info[g, 3])]
                '''

        gps_info = np.asarray(info['gps'], dtype=np.float32)
        assert gps_info.shape[0] == len(img_dir_list)
        '''
                shape [N, 3], N is equal to the number of images.

                gps_info[i, :] is a GPS mapping record of a pedestrian image. 
                This is obtained by mapping the coordinates between image coordinates and world coordinates.

                gps_info[i, 0] if the timestamp of image img_dir_list[i]
                gps_info[i, 1] if the GPS longitude of image img_dir_list[i]
                gps_info[i, 2] if the GPS latitude of image img_dir_list[i]
                '''

        trajectory_info = info['trajectory']
        '''
                type dict
                trajectory_info records the GPS location of the pedestrian's mobile phone. 
                This is the wireless signal information.
                
                trajectory_info.keys() is the list of person ID of the pedestrian with mobile phone.
                trajectory_info[person_ID] is a GPS location record of a pedestrian's mobile phone 
                during the entire data collection process.
                
                typoe of trajectory_info[person_ID] is np.ndarray
                shape of trajectory_info[person_ID] is [T, 3]

                trajectory_info[person_ID][i, 0] if the timestamp
                trajectory_info[person_ID][i, 1] if the GPS longitude
                trajectory_info[person_ID][i, 2] if the GPS latitude
                '''

        new_trajectory_info = {}
        p_ids = np.unique(gallery_info[:, 0]).tolist()
        for k, v in trajectory_info.items():
            assert k in p_ids
            new_trajectory_info[k] = np.asarray(v, np.float32)

        data_dict = {}
        data_dict['dir'] = img_dir_list
        data_dict['probe'] = probe_info
        data_dict['gallery'] = gallery_info
        data_dict['info'] = 'WPReID dataset'
        data_dict['gps'] = gps_info
        data_dict['trajectory'] = new_trajectory_info

        return data_dict
