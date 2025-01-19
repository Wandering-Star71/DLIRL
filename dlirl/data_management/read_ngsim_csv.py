import numpy as np
import pandas as pd


def read_track_csv(arguments):
    df = pd.read_csv(arguments["input_path"])
    grouped = df.groupby(['Vehicle_ID'], sort=False)
    frame_grouped = df.groupby(['Frame_ID'], sort=False)
    tracks = {}
    static_info = {}
    for group_id, rows in grouped:
        tracks[np.int64(group_id[0])] = {
            'Frame_ID': np.int64(rows['Frame_ID'].values),
            'Local_X': rows['Local_Y'].values * 0.3048,
            'Local_Y': rows['Local_X'].values * 0.3048,
            'v_Vel': rows['v_Vel'].values * 0.3048,
            'v_Acc': rows['v_Acc'].values * 0.3048,
            'Lane_ID': np.int64(rows['Lane_ID'].values),
            # 'Preceeding': np.int64(rows['Preceding'].values),
            # 'Following': np.int64(rows['Following'].values),
            # 'Space_Hdwy': rows['Space_Headway'].values,
            # 'Time_Hdwy': rows['Time_Headway'].values
        }
        static_info[np.int64(group_id[0])] = {
            'Total_Frames': len(np.int64(rows['Total_Frames'].values)),
            'v_Length': rows['v_Length'].values[0] * 0.3048,
            'v_Width': rows['v_Width'].values[0] * 0.3048,
            'driving_direction': 1,
            'initial_frame': np.int64(rows['Frame_ID'].values[0]),
        }
        # static_info[track_id]
    return tracks, static_info, frame_grouped
