import tqdm
import math
import os
import shutil
import jax.numpy as jnp
from dlirl.data_management.read_ngsim_csv import *
from dlirl.part import Transition
from dlirl.util import save_transition_to_disk

path_ori = ["../NGSIMRawData/trajectories-0400-0415.csv",
            "../NGSIMRawData/trajectories-0500-0515.csv",
            "../NGSIMRawData/trajectories-0515-0530.csv",
            "../NGSIMRawData/trajectories-0750am-0805am.csv",
            "../NGSIMRawData/trajectories-0805am-0820am.csv",
            "../NGSIMRawData/trajectories-0820am-0835am.csv",
            ]
path_output = '../NGSIMRawData/transition/'
delta_t = 0.1  # s


def generate_sequences(length):
    sequences = []
    start = 0
    while start + 6 <= length:
        end = min(start + 2 * 3, length)
        sequences.append([i for i in range(start, end + 1) if i % 2 == 0])
        start += 12
    return sequences


def main():
    transition_count = 0
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    else:
        shutil.rmtree(path_output)
        os.makedirs(path_output)
    for idx, path in enumerate(path_ori):
        if not os.path.exists(path_output + '{}/'.format(idx + 1)):
            os.makedirs(path_output + '{}/'.format(idx + 1))
        args = {'input_path': path}
        track_info, static_info, frame_grouped = read_track_csv(args)
        pbar = tqdm.tqdm(static_info.keys())
        for veh_index in pbar:
            veh_track = track_info[veh_index]
            veh_info = static_info[veh_index]
            veh_driving_dir = veh_info['driving_direction']

            samples = generate_sequences(veh_info['Total_Frames'] - 3)
            for sample in samples:
                state_spec = []
                action_spec = []
                for veh_id_run_time in sample:
                    veh_global_frame = veh_track['Frame_ID'][veh_id_run_time]
                    veh_lane_id = veh_track['Lane_ID'][veh_id_run_time]

                    veh_x_t = veh_track['Local_X'][veh_id_run_time]
                    veh_y_t = veh_track['Local_Y'][veh_id_run_time]
                    veh_x_tp1 = veh_track['Local_X'][veh_id_run_time + 1]
                    veh_y_tp1 = veh_track['Local_Y'][veh_id_run_time + 1]
                    veh_x_tp2 = veh_track['Local_X'][veh_id_run_time + 2]
                    veh_y_tp2 = veh_track['Local_Y'][veh_id_run_time + 2]
                    veh_v_angle_t = math.atan2(veh_y_tp1 - veh_y_t, veh_x_tp1 - veh_x_t)
                    # veh_test_v_t = math.sqrt((veh_y_tp1 - veh_y_t)**2 + (veh_x_tp1 - veh_x_t)**2) / delta_t
                    veh_v_abs_t = veh_track['v_Vel'][veh_id_run_time]
                    veh_vx_t = veh_v_abs_t * math.cos(veh_v_angle_t)
                    veh_vy_t = veh_v_abs_t * math.sin(veh_v_angle_t)
                    veh_v_angle_tp1 = math.atan2(veh_y_tp2 - veh_y_tp1, veh_x_tp2 - veh_x_tp1)
                    veh_v_abs_tp1 = veh_track['v_Vel'][veh_id_run_time + 1]
                    veh_vx_tp1 = veh_v_abs_tp1 * math.cos(veh_v_angle_tp1)
                    veh_vy_tp1 = veh_v_abs_tp1 * math.sin(veh_v_angle_tp1)
                    veh_ax_t = (veh_vx_tp1 - veh_vx_t) / delta_t
                    veh_ay_t = (veh_vy_tp1 - veh_vy_t) / delta_t
                    state_box = [veh_vx_t, veh_vy_t, veh_ax_t, veh_ay_t, -1, -1, -1, -1, -1, -1, -1,
                                 -1]  # here is the state

                    fg = frame_grouped.get_group((veh_global_frame,))
                    sv_id_list = [np.int64(id) for global_frame, id in fg['Vehicle_ID'].items()]
                    sv_id_list.remove(veh_index)

                    if sv_id_list is not None:
                        for sv_id in sv_id_list:
                            sv_info = static_info[sv_id]
                            sv_track = track_info[sv_id]
                            sv_driving_dir = sv_info['driving_direction']
                            sv_run_time = veh_global_frame - sv_info['initial_frame']
                            if sv_driving_dir != veh_driving_dir:
                                # print('vehicle {} \'s {} frame skipped'.format(veh_index, veh_id_run_time))
                                continue
                            if sv_run_time + 2 >= sv_info['Total_Frames']:
                                continue
                            sv_lane_id = sv_track['Lane_ID'][sv_run_time]

                            sv_x_t = sv_track['Local_X'][sv_run_time]
                            sv_y_t = sv_track['Local_Y'][sv_run_time]
                            sv_x_tp1 = sv_track['Local_X'][sv_run_time + 1]
                            sv_y_tp1 = sv_track['Local_Y'][sv_run_time + 1]
                            sv_v_angle_t = math.atan2(sv_y_tp1 - sv_y_t, sv_x_tp1 - sv_x_t)
                            sv_v_abs_t = sv_track['v_Vel'][sv_run_time]
                            sv_vx_t = sv_v_abs_t * math.cos(sv_v_angle_t)
                            sv_vy_t = sv_v_abs_t * math.sin(sv_v_angle_t)
                            delta_d = math.sqrt((sv_x_t - veh_x_t) ** 2 + (sv_y_t - veh_y_t) ** 2)
                            delta_v = math.sqrt((sv_vx_t - veh_vx_t) ** 2 + (sv_vy_t - veh_vy_t) ** 2 + 1e-3)
                            ttc = delta_d / delta_v * veh_driving_dir

                            if (sv_lane_id - veh_lane_id) * veh_driving_dir == 0:  # center
                                if (sv_x_t - veh_x_t) * veh_driving_dir > 0:  # front
                                    if abs(state_box[4]) > abs(ttc): state_box[4] = ttc
                                else:  # behind
                                    if abs(state_box[5]) > abs(ttc): state_box[5] = ttc
                                pass
                            elif (sv_lane_id - veh_lane_id) * veh_driving_dir == 1:  # right
                                if (sv_x_t - veh_x_t) * veh_driving_dir > sv_info['v_Length']:  # front
                                    if abs(state_box[9]) > abs(ttc): state_box[9] = ttc
                                elif (sv_x_t - veh_x_t) * veh_driving_dir < -sv_info['v_Length']:  # behind
                                    if abs(state_box[11]) > abs(ttc): state_box[11] = ttc
                                else:  # middle
                                    if abs(state_box[10]) > abs(ttc): state_box[10] = ttc
                            elif (sv_lane_id - veh_lane_id) * veh_driving_dir == -1:  # left
                                if (sv_x_t - veh_x_t) * veh_driving_dir > sv_info['v_Length']:  # front
                                    if abs(state_box[6]) > abs(ttc): state_box[6] = ttc
                                elif (sv_x_t - veh_x_t) * veh_driving_dir < -sv_info['v_Length']:  # behind
                                    if abs(state_box[8]) > abs(ttc): state_box[8] = ttc
                                else:  # middle
                                    if abs(state_box[7]) > abs(ttc): state_box[7] = ttc
                            else:  # 2 lane alongside
                                pass
                            # if veh_index%2 == 1:
                            #     for i in range(4, 11): state_box[i] *= -1
                    else:
                        # print('vehicle {} \'s {} frame skipped'.format(veh_index, veh_id_run_time))
                        pass
                    state_spec.append(state_box)
                    action_spec = [veh_ax_t, veh_ay_t * 10]
                transition = Transition(state=jnp.array(state_spec),
                                        action=jnp.array(action_spec))
                transition_count += 1
                fname = path_output + '{}/{}.transition'.format(idx + 1, transition_count)
                save_transition_to_disk(transition, fname)


if __name__ == "__main__":
    main()