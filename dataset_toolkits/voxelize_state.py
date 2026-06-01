import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d
import json

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 1,
        "drawer": 2,
        "base": 3,
        "handle": 4,
        "wheel": 5,
        "knob": 6,
        "shelf": 7,
        "tray": 8
    },
    "bwd": {
        1: "door",
        2: "drawer",
        3: "base",
        4: "handle",
        5: "wheel",
        6: "knob",
        7: "shelf",
        8: "tray"
    }
}

# reference of joint types for each part
joint_ref = {
    "fwd": {
       "fixed": 1,
        "revolute": 2,
        "prismatic": 3,
        "screw": 4,
        "continuous": 5 
    },
    "bwd": {
        1: "fixed",
        2: "revolute",
        3: "prismatic",
        4: "screw",
        5: "continuous"
    } 
}

def load_articulation(articulation_path):
    articulation_list = []
    with open(articulation_path, 'r') as f:
        articulation = json.load(f)
        tree = articulation['diffuse_tree']
        root_id = 0
        for node in tree:
            if node['parent'] == -1:
                root_id = node['id']
        for node in tree:
            joint_type = np.array([joint_ref['fwd'][node['joint']['type']]], dtype=np.float32)
            # 对于 fixed 关节，axis 和 range 都用 0 填充
            if node['joint']['type'] == 'fixed':
                axis_d = np.zeros(3, dtype=np.float32)
                axis_o = np.zeros(3, dtype=np.float32)
                jrange = np.zeros(2, dtype=np.float32)
            else:
                axis_d = np.array(node['joint']['axis']['direction'], dtype=np.float32)
                axis_o = np.array(node['joint']['axis']['origin'], dtype=np.float32)
                jrange = np.array(node['joint']['range'], dtype=np.float32)

            if node['id'] == root_id or node['parent'] == root_id:
                node_data = np.concatenate([axis_d, axis_o, jrange, joint_type], axis=0)
            else:
                parent_id = node['parent']
                parent_node = next(x for x in tree if x['id'] == parent_id)
                if parent_node['joint']['type'] == 'fixed':
                    axis_d_p = np.zeros(3, dtype=np.float32)
                    axis_o_p = np.zeros(3, dtype=np.float32)
                    jrange_p = np.zeros(2, dtype=np.float32)
                else:
                    axis_d_p = np.array(parent_node['joint']['axis']['direction'], dtype=np.float32)
                    axis_o_p = np.array(parent_node['joint']['axis']['origin'], dtype=np.float32)
                    jrange_p = np.array(parent_node['joint']['range'], dtype=np.float32)
                jtype_p = parent_node['joint']['type']
                jtype_p_id = np.array([joint_ref['fwd'][jtype_p]], dtype=np.float32)
                node_data = np.concatenate([axis_d_p, axis_o_p, jrange_p, jtype_p_id], axis=0)
            articulation_list.append(node_data)
    return articulation_list

if __name__ == '__main__':
    sha256='000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10'

    # get file list by loading the data_split_dyn_slat.json
    with open('/mnt/slurm_home/hhchen/cageplus/indexes/data_split_dyn_slat_all_0507.json', 'r') as f:
        data_split = json.load(f)
    train_objs = data_split['train']
    all_output_dir = []
    for obj in train_objs:
        output_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/voxels_state_1000000"
        os.makedirs(output_dir, exist_ok=True)
        # all_output_dir.append(output_dir)

        # 1. load labeled voxels
        data_path = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global"
        label_path = os.path.join(data_path, 'voxels_label_1000000', f'{sha256}.txt')
        if os.path.exists(label_path):
            labeled_voxels = np.loadtxt(label_path)
        else:
            print(f"label_path not found for {obj}, skipping")
            continue

        # 2. get voxel belongs to different parts
        part_indices = labeled_voxels[:, -1].astype(np.int32)
        part_indices_unique = np.unique(part_indices)
        part_voxels_id_list = []
        for part_index in part_indices_unique:
            part_voxels_id = np.where(part_indices == part_index)[0]  # Get indices where condition is True
            part_voxels_id_list.append(part_voxels_id)
            
        # 3. load articulation state
        obj_json_path = os.path.join('/mnt/slurm_home/hhchen/cageplus/data', obj, 'train_slat32_blender.json')
        if os.path.exists(obj_json_path):
            articulation_list = load_articulation(obj_json_path)
            
            # 4. articulate voxels for each part based on joint parameters
            num_states = 8  # You can adjust this value as needed
            for state in range(num_states):
                state_voxels = []
                for part_idx, part_voxels_id in enumerate(part_voxels_id_list):
                    # get part voxels
                    part_voxels = labeled_voxels[part_voxels_id, :3]  # get xyz coordinates
                    
                    # get articulation parameters
                    articulation = articulation_list[part_idx]
                    axis_d = articulation[0:3]
                    axis_o = articulation[3:6]
                    jrange = articulation[6:8]
                    jtype = int(articulation[8])
                    
                    # create transformation matrix
                    if state == 0:
                        transform = np.eye(4)
                    else:
                        transform = np.eye(4)
                        if jtype == joint_ref['fwd']['fixed']:
                            pass
                        elif jtype == joint_ref['fwd']['revolute']:
                            # normalize axis direction
                            axis = axis_d / np.linalg.norm(axis_d)
                            angle_rad = np.deg2rad(jrange[1] / num_states)
                            # create rotation matrix around axis
                            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
                            transform[:3, :3] = R
                            # apply translation for rotation center
                            T1 = np.eye(4)
                            T1[:3, 3] = -axis_o
                            T2 = np.eye(4)
                            T2[:3, 3] = axis_o
                            transform = T2 @ transform @ T1
                        elif jtype == joint_ref['fwd']['prismatic']:
                            # normalize axis direction
                            axis = axis_d / np.linalg.norm(axis_d)
                            distance = jrange[1] / num_states 
                            transform[:3, 3] = axis * distance
                        elif jtype == joint_ref['fwd']['screw']:
                            axis = axis_d / np.linalg.norm(axis_d)
                            dist = jrange[0] / num_states
                            theta = np.deg2rad(jrange[1] / num_states)
                            # rotation
                            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
                            transform[:3, :3] = R
                            # translation
                            T1 = np.eye(4)
                            T1[:3, 3] = -axis_o
                            T2 = np.eye(4)
                            T2[:3, 3] = axis_o
                            T3 = np.eye(4)
                            T3[:3, 3] = axis * dist
                            transform = T3 @ T2 @ transform @ T1
                        elif jtype == joint_ref['fwd']['continuous']:
                            axis = axis_d / np.linalg.norm(axis_d)
                            theta = np.deg2rad(jrange[1] / num_states)
                            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
                            transform[:3, :3] = R
                            T1 = np.eye(4)
                            T1[:3, 3] = -axis_o
                            T2 = np.eye(4)
                            T2[:3, 3] = axis_o
                            transform = T2 @ transform @ T1
                    
                    # apply transformation to voxels
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(part_voxels)
                    pcd.transform(transform)
                    transformed_voxels = np.asarray(pcd.points)
                    
                    # store transformed voxels with part label
                    # part_labels = np.ones((transformed_voxels.shape[0], 1)) * part_idx
                    # transformed_voxels_with_label = np.hstack([transformed_voxels, part_labels])
                    # state_voxels.append(transformed_voxels_with_label)
                    labeled_voxels[part_voxels_id, :3] = transformed_voxels
                
                # save state voxels
                np.savetxt(os.path.join(output_dir, f'{sha256}_{state}.txt'), labeled_voxels)
                print(f"Done for {obj}")
        else:
            print(f"obj_json_path not found for {obj}, skipping")
            continue
    

    
    
