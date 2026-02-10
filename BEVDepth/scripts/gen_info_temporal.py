import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm


def _pack_sample_data_info(nusc, sd):
    """sd: sample_data record"""
    info = {
        "sample_token": sd["sample_token"],
        "timestamp": sd["timestamp"],
        "is_key_frame": sd.get("is_key_frame", False),
        "filename": sd["filename"],
        "ego_pose": nusc.get("ego_pose", sd["ego_pose_token"]),
        "calibrated_sensor": nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]),
    }
    # camera-only fields exist on camera sample_data
    if "height" in sd:
        info["height"] = sd["height"]
    if "width" in sd:
        info["width"] = sd["width"]
    return info


def _collect_prev_sweeps(nusc, start_sd, max_sweeps):
    """
    start_sd: current(key) sample_data record
    returns list of length <= max_sweeps containing previous sample_data infos
    (t-1, t-2, ...)
    """
    sweeps = []
    cur = start_sd
    for _ in range(max_sweeps):
        if cur["prev"] == "":
            break
        cur = nusc.get("sample_data", cur["prev"])
        sweeps.append(_pack_sample_data_info(nusc, cur))
    return sweeps


def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=10):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue

        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)

        while True:
            info = dict()

            # ------------------------------------------------------------------
            # [ADD] sample-level temporal linkage (핵심)
            # ------------------------------------------------------------------
            info['sample_token'] = cur_sample['token']
            info['prev_sample_token'] = cur_sample['prev']    
            info['next_sample_token'] = cur_sample['next']    # (optional)
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            # ------------------------------------------------------------------

            cam_datas = list()
            lidar_datas = list()

            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']

            cam_infos = dict()
            lidar_infos = dict()

            # --------------------
            # Camera key-frame info
            # --------------------
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data', cur_sample['data'][cam_name])
                cam_datas.append(cam_data)

                sweep_cam_info = dict(
                    sample_token=cam_data['sample_token'],
                    ego_pose=nusc.get('ego_pose', cam_data['ego_pose_token']),
                    timestamp=cam_data['timestamp'],
                    is_key_frame=cam_data['is_key_frame'],
                    height=cam_data['height'],
                    width=cam_data['width'],
                    filename=cam_data['filename'],
                    calibrated_sensor=nusc.get(
                        'calibrated_sensor',
                        cam_data['calibrated_sensor_token']
                    ),
                )
                cam_infos[cam_name] = sweep_cam_info

            # --------------------
            # LiDAR key-frame info
            # --------------------
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)

                sweep_lidar_info = dict(
                    sample_token=lidar_data['sample_token'],
                    ego_pose=nusc.get('ego_pose',
                                      lidar_data['ego_pose_token']),
                    timestamp=lidar_data['timestamp'],
                    filename=lidar_data['filename'],
                    calibrated_sensor=nusc.get(
                        'calibrated_sensor',
                        lidar_data['calibrated_sensor_token']
                    ),
                )
                lidar_infos[lidar_name] = sweep_lidar_info

            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            ann_infos = list()
            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos



def main():
    trainval_nusc = NuScenes(version="v1.0-trainval", dataroot="./data/nuScenes/", verbose=True)
    train_infos = generate_info(trainval_nusc, splits.train)
    val_infos = generate_info(trainval_nusc, splits.val)
    mmcv.dump(train_infos, "./data/nuScenes/nuscenes_infos_train_temporal.pkl")
    mmcv.dump(val_infos, "./data/nuScenes/nuscenes_infos_val_temporal.pkl")

    test_nusc = NuScenes(version="v1.0-test", dataroot="./data/nuScenes/", verbose=True)
    test_infos = generate_info(test_nusc, splits.test)
    mmcv.dump(test_infos, "./data/nuScenes/nuscenes_infos_test_temporal.pkl")


if __name__ == "__main__":
    main()
