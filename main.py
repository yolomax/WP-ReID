from dataset import DataBank
from pathlib import Path
from utils import file_abs_path, DataPacker
from eval import update_with_gps

if __name__ == '__main__':
    files_dir = file_abs_path(__file__) / 'files'
    mmt_duke_dist_file = files_dir / 'mmt_duke_old_g2g_distmat.json'

    wp_reid_dataset = DataBank(minframes=3)
    distmat = DataPacker.load(mmt_duke_dist_file)['g2g_distmat']
    update_with_gps(distmat, wp_reid_dataset.test_info, wp_reid_dataset.probe_index, wp_reid_dataset.gallery_index,
                    wp_reid_dataset.gps_info, wp_reid_dataset.trajectory_info)



