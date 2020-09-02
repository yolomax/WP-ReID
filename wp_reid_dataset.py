from pathlib import Path
from utils import DataPacker


class WPReID(object):
    def __init__(self):
        root_dir = Path('/data/liuyh/data/WPReID')
        self.raw_data_folder = root_dir / 'WPReID'
        self.cropped_images_dir = self.raw_data_folder / 'cropped_data'

        self.split_file_dir = self.raw_data_folder / 'wp_reid_info.json'

        self.all_image_num = 106578
        self.resize_hw = (256, 128)

        self.check_raw_file()

    def check_raw_file(self):
        assert self.raw_data_folder.exists()
        assert self.split_file_dir.exists()

    def get_dict(self):

        info = DataPacker.load(self.split_file_dir)
        img_dir_list = info['img_dir']
        img_dir_list = [str(self.raw_data_folder / i) for i in img_dir_list]
        probe_info = info['probe']
        gallery_info = info['gallery']
        gps_info = info['gps']
        trajectory_info = info['trajectory']

        data_dict = {}
        data_dict['dir'] = img_dir_list
        data_dict['probe'] = probe_info
        data_dict['gallery'] = gallery_info
        data_dict['info'] = 'WPReID dataset'
        data_dict['gps'] = gps_info
        data_dict['info'] = trajectory_info

        return data_dict
