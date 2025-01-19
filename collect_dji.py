import os
import shutil
import tqdm
import math
import jax.numpy as jnp
from typing import NamedTuple
import pickle5 as pickle
from dlirl.data_management.read_dji_csv import *
from dlirl.part import Transition

HIGHD_DATA_PATH = '../DJIRawData/'
DATA_PATH = '../DJIRawData/transition/'
# DJI fps is 30
def generate_sequences(length):
    sequences = []
    start = 0
    while start + 20 <= length:
        end = start + 6 * 3
        sequences.append([i for i in range(start, end + 1) if i % 6 == 0])
        start += 42
    return sequences
def main():
    here = os.path.abspath(os.path.dirname(__file__))
    ROLLOUT_MAIN_PATH = os.path.join(here, DATA_PATH)
    if not os.path.exists(ROLLOUT_MAIN_PATH):
        os.makedirs(ROLLOUT_MAIN_PATH)
    else:
        shutil.rmtree(ROLLOUT_MAIN_PATH)
        os.makedirs(ROLLOUT_MAIN_PATH)
    for file_index in range(1, 69):
        if file_index < 10:
            no_str = '0{}'.format(file_index)
        else:
            no_str = str(file_index)
        args = {'input_path': HIGHD_DATA_PATH + 'DJI_00{}/{}_tracks.csv'.format(no_str, no_str),
                'input_static_path': HIGHD_DATA_PATH + 'DJI_00{}/{}_tracksMeta.csv'.format(no_str, no_str),
                'input_meta_path': HIGHD_DATA_PATH + 'DJI_00{}/{}_recordingMeta.csv'.format(no_str, no_str),
                # 'pickle_path': HIGHD_DATA_PATH + '{}.pickle'.format(no_str),
                # 'background_image': HIGHD_DATA_PATH + '{}_highway.png'.format(no_str)
                }
        track_info = read_track_csv(args)
        static_info = read_static_info(args)
        # meta_dictionary = read_meta_info(args)

        ROLLOUT_FILE_PATH = ROLLOUT_MAIN_PATH + '{}/'.format(file_index)
        if not os.path.exists(ROLLOUT_FILE_PATH):
            os.makedirs(ROLLOUT_FILE_PATH)

        fname_count = 0
        pbar = tqdm.tqdm(static_info.keys())
        for veh_index in pbar:
            pbar.set_description('file {}'.format(file_index))
            veh_info = static_info[veh_index]
            veh_track = track_info[veh_index]

            veh_driving_dir = -1 if veh_info["drivingDirection"] == 1 else 1 # 下1上-1
            samples = generate_sequences(veh_info['numFrames']-1)
            for sample in samples:
                veh_frame_start = veh_info['initialFrame']
                state_spec = []
                for veh_run_time in sample:
                    state_bbox = []
                    state_bbox.extend([veh_track['xVelocity'][veh_run_time] * veh_driving_dir,
                                       veh_track['yVelocity'][veh_run_time] * veh_driving_dir,
                                       veh_track['xAcceleration'][veh_run_time] * veh_driving_dir,
                                       veh_track['yAcceleration'][veh_run_time] * veh_driving_dir])
                    # move forward: positive x, movw left: positive y

                    sv_list = [veh_track["precedingId"][veh_run_time],
                               veh_track["followingId"][veh_run_time],
                               veh_track["leftPrecedingId"][veh_run_time],
                               veh_track["leftAlongsideId"][veh_run_time],
                               veh_track["leftFollowingId"][veh_run_time],
                               veh_track["rightPrecedingId"][veh_run_time],
                               veh_track["rightAlongsideId"][veh_run_time],
                               veh_track["rightFollowingId"][veh_run_time]]
                    for sv_index in sv_list:
                        if sv_index != 0:
                            sv_track = track_info[sv_index]
                            sv_info = static_info[sv_index]
                            if (sv_info['initialFrame'] <= veh_run_time + veh_frame_start
                                    <= sv_info['finalFrame']):
                                sv_run_time = veh_run_time + veh_frame_start - sv_info['initialFrame']
                                delta_d = math.sqrt(
                                    (sv_track['bbox'][sv_run_time][0] - veh_track['bbox'][veh_run_time][0]) ** 2
                                    + (sv_track['bbox'][sv_run_time][1] - veh_track['bbox'][veh_run_time][1]) ** 2)
                                delta_v = math.sqrt(
                                    (sv_track['xVelocity'][sv_run_time] - veh_track['xVelocity'][veh_run_time]) ** 2
                                    + (sv_track['yVelocity'][sv_run_time] - veh_track['yVelocity'][veh_run_time]) ** 2
                                    + 1e-3)
                                ttc = delta_d / delta_v
                            else:
                                ttc = -1
                        else:
                            ttc = -1
                        state_bbox.append(ttc * veh_driving_dir)
                    assert len(state_bbox) == 12
                    state_spec.append(state_bbox)
                acceleration_x = veh_track['xAcceleration'][sample[-1]] * veh_driving_dir
                acceleration_y = veh_track['yAcceleration'][sample[-1]] * veh_driving_dir * 10
                action_spec = jnp.array([acceleration_x, acceleration_y])
                state_spec = jnp.array(state_spec)

                transition = Transition(state=state_spec,
                                        action=action_spec)
                fname = ROLLOUT_FILE_PATH + '{}.transition'.format(fname_count)
                fname_count += 1
                with open(fname, 'wb') as f:
                    pickle.dump(transition, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()