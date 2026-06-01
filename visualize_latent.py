import os
import sys
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis import models as models
from trellis.modules import sparse as sp
from trellis.utils import render_utils
import imageio
import json
from cageplus.utils.refs import sem_ref, joint_ref
# from cageplus.utils.render import get_rotation_axis_angle

torch.set_grad_enabled(False)
anim_count = 100

def get_rotation_axis_angle(k, theta):
    '''
    Rotation matrix converter from axis-angle using Rodrigues' rotation formula

    Args:
        k (np.ndarray): 3D unit vector representing the axis to rotate about.
        theta (float): Angle to rotate with in radians.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def Transform_GS(gs, radius=0.01, jtype=None, jrange=None, axis_d=None, axis_o=None):
    '''
    Function to get the transformed gaussian

    Args:
    - center (np.array): bounding box center
    - size (np.array): bounding box size
    - radius (float): radius of the cylinder
    - jtype (int): joint type
    - jrange (list): joint range
    - axis_d (np.array): axis direction
    - axis_o (np.array): axis origin

    Returns:
    - gs_anim (gs object): gs object at opening state
    '''

    # pcd_box = o3d.geometry.PointCloud()
    # pcd_box.points = o3d.utility.Vector3dVector(pcd)

    # transform
    gs_anim = copy.deepcopy(gs)
    if jtype == 2: # revolute
        # import pdb; pdb.set_trace()
        theta = np.deg2rad(jrange[1]/2)
        gs_anim.translate_gs(-axis_o)
        R = get_rotation_axis_angle(-axis_d, theta)
        gs_anim.transform_gs(R)
        gs_anim.translate_gs(axis_o)
    elif jtype == 3: # prismatic
        dist = np.array(jrange[1])
        gs_anim.translate_gs(-axis_d * dist)
    elif jtype == 4: # screw
        dist = np.array(jrange[1])
        theta = 0.25 * np.pi
        R = get_rotation_axis_angle(-axis_d, theta)
        gs_anim.translate_gs(-axis_o)
        gs_anim.transform_gs(R)
        gs_anim.translate_gs(axis_o)
        gs_anim.translate_gs(-axis_d * dist)
    elif jtype == 5: # continuous
        theta = 0.25 * np.pi
        R = get_rotation_axis_angle(-axis_d, theta)
        gs_anim.translate_gs(-axis_o)
        gs_anim.transform_gs(R)
        gs_anim.translate_gs(axis_o)
    
    return gs_anim

def Transform_GS_list(gs, radius=0.01, jtype=None, jrange=None, axis_d=None, axis_o=None):
    '''
    Function to get the transformed gaussian

    Args:
    - center (np.array): bounding box center
    - size (np.array): bounding box size
    - radius (float): radius of the cylinder
    - jtype (int): joint type
    - jrange (list): joint range
    - axis_d (np.array): axis direction
    - axis_o (np.array): axis origin

    Returns:
    - gs_anim (gs object): gs object at opening state
    '''

    # pcd_box = o3d.geometry.PointCloud()
    # pcd_box.points = o3d.utility.Vector3dVector(pcd)

    # transform
    # gs_anim = copy.deepcopy(gs)
    gs_anim_list = []
    for i in range(anim_count):
        gs_anim = copy.deepcopy(gs)
        if jtype == 2: # revolute
            # import pdb; pdb.set_trace()
            theta = np.deg2rad(0+jrange[1]/20*(i-1))
            gs_anim.translate_gs(-axis_o)
            R = get_rotation_axis_angle(-axis_d, theta)
            gs_anim.transform_gs(R)
            gs_anim.translate_gs(axis_o)
        elif jtype == 3: # prismatic
            dist = np.array(jrange[1])
            gs_anim.translate_gs(-axis_d * dist)
        elif jtype == 4: # screw
            dist = np.array(jrange[1])
            theta = 0.25 * np.pi
            R = get_rotation_axis_angle(-axis_d, theta)
            gs_anim.translate_gs(-axis_o)
            gs_anim.transform_gs(R)
            gs_anim.translate_gs(axis_o)
            gs_anim.translate_gs(-axis_d * dist)
        elif jtype == 5: # continuous
            theta = 0.25 * np.pi
            R = get_rotation_axis_angle(-axis_d, theta)
            gs_anim.translate_gs(-axis_o)
            gs_anim.transform_gs(R)
            gs_anim.translate_gs(axis_o)

        gs_anim_list.append(gs_anim)
    
    return gs_anim_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None, help='Filter objects with low aesthetic scores')
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg', help='Feature model')
    parser.add_argument('--enc_pretrained', type=str, default='JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results', help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None, help='Encoder model, if specified, overrides pretrained model')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None, help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))


    records = []
    # import pdb; pdb.set_trace()


    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    decoder = pipeline.models

    def get_sha256_list(feature_dir):

        filenames = os.listdir(feature_dir)

        sha256_list = [os.path.splitext(filename)[0] for filename in filenames]
    
        return sha256_list
    
    # Define the directory path
    directory_path = '/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results'
    # import pdb; pdb.set_trace()

    # Check if the directory exists and create it if it doesn't
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    # 逐个处理
    filelists = get_sha256_list(os.path.join(opt.output_dir, 'latents', opt.feat_model+'_slat_enc_swin8_B_64l8_fp16', ))
    # for sha256 in tqdm(filelists, desc="Processing"):
    gaussian_list = []
    gaussian_list_list = []
    obj_json_path = '/mnt/slurm_home/hhchen/cageplus/data/Refrigerator/10685/train_slat32.json'
    with open(obj_json_path, 'r') as f:
        obj_json = json.load(f)
        tree = obj_json['diffuse_tree']
        K = len(tree)

        root_id = 0
        for node in tree:
            if node['parent'] == -1:
                root_id = node['id'] 

        for i in range(K):
            # decode sparse structure
            # z_s = np.load((f'/mnt/slurm_home/hhchen/cageplus/train_data_split_p6_slat_x_attr_8norm/cage/demo/images/predict/cond_graph/07499/Table/c093e58644897609c1d1671be52335e5/val_slat_0_{i}.npz'), allow_pickle=True)
            # z_s_tensor = torch.from_numpy(z_s['mean']).cuda().unsqueeze(0)

            # #######################################visualize the sparse structure#######################################
            # # import numpy as np
            # # import matplotlib.pyplot as plt
            # # from sklearn.manifold import TSNE
            # # from sklearn.decomposition import PCA

            # # # 展平: (16,16,16,8) -> (4096, 8)
            # # latent_feature = z_s_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            # # flattened = latent_feature.reshape(-1, 8)

            # # # PCA 降到 3 维
            # # pca = PCA(n_components=3)
            # # reduced_features = pca.fit_transform(flattened)

            # # # 生成 3D 坐标 (xyz)
            # # xyz = np.array(np.meshgrid(np.arange(8), np.arange(8), np.arange(8))).T.reshape(-1, 3)

            # # # 3D 可视化
            # # fig = plt.figure(figsize=(8, 8))
            # # ax = fig.add_subplot(111, projection='3d')
            # # sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=reduced_features[:, 0], cmap='viridis', s=5)

            # # plt.colorbar(sc, label="PCA Component 1")
            # # plt.title("PCA Reduced 3D Feature Visualization")

            # # # **保存图片**
            # # plt.savefig("pca_3d.png", dpi=300, bbox_inches='tight')
            # # plt.close()  # 关闭窗口，防止服务器挂起

            # # data_to_save = np.hstack([xyz, reduced_features])  # (4096, 4)

            # # # 保存为 TXT 文件
            # # np.savetxt("xyz_pca1.txt", data_to_save, fmt="%.6f", delimiter=" ", comments='')

            # # import pdb; pdb.set_trace()
            # z_s_tensor = z_s_tensor * 6 #(0.41863010797458583/z_s_tensor.std())
            # z_s_tensor = z_s_tensor.to(dtype=torch.float32)  # Convert to single precision if needed
            # decoder2 = pipeline.models['sparse_structure_decoder']
            # coords = torch.argwhere(decoder2(z_s_tensor)>0)[:, [0, 2, 3, 4]].int()
            # # save coords to txt
            # np.savetxt(f'/mnt/slurm_home/hhchen/cageplus/train_data_split_p6_slat_x_attr/cage/demo/images/predict/cond_graph/07999/Safe/36d1bd41675efbce9b1be43bfefc8ea6/val_voxels_{i}.txt', coords.cpu().numpy(), fmt='%d')

            # feats = np.load(os.path.join(opt.output_dir, 'latents', opt.feat_model+'_slat_enc_swin8_B_64l8_fp16', f'{sha256}.npz'), allow_pickle=True)
            # latent = sp.SparseTensor(feats = torch.from_numpy(feats['feats']).float(), coords = torch.cat([torch.zeros(feats['feats'].shape[0], 1).int(),torch.from_numpy(feats['coords']).int(),], dim=1),).cuda()
            # import pdb; pdb.set_trace()

            #######################################decode the slat to GS#######################################
            feats = np.load(f'/mnt/slurm_home/hhchen/cageplus/train_data_split_p6_slat_x_attr_8norm/cage/demo/images/predict/cond_graph/07499/Refrigerator/c093e58644897609c1d1671be52335e5/GT_slat_{i}.npz', allow_pickle=True)   
            coords = feats['coords']
            # np.savetxt(os.path.join(f"/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/{i}_coords.txt"), coords, fmt='%d')
            latent = sp.SparseTensor(feats = torch.from_numpy(feats['feats']).float(), coords = torch.cat([torch.zeros(feats['feats'].shape[0], 1).int(),torch.from_numpy(coords).int(),], dim=1),).cuda()

            ret = {}
            ret['gaussian'] = decoder['slat_decoder_gs'](latent)
            ret['gaussian'][0].transform_gs()

            # ret['radiance_field'] = decoder['slat_decoder_rf'](latent)
            # ret['mesh'] = decoder['slat_decoder_mesh'](latent)

            # transform to the bbox
            # load obj_json_path
            node = tree[i]
            aabb_center = np.array(node['aabb']['center'], dtype=np.float32) # (3,), range from -1 to 1
            aabb_center = torch.from_numpy(aabb_center).cuda().unsqueeze(0)

            blender_render_json = f"/mnt/slurm_home/hhchen/cageplus/data/Refrigerator/10685/{i}/renders/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10/transforms.json"
            # load json
            with open(blender_render_json, 'r') as f:
                blender_render = json.load(f)
                # get the scale and offset
                blender_scale = blender_render['scale']
                blender_offset = blender_render['offset']

            # ret['gaussian'][0].transform_gs()
            ret['gaussian'][0].save_ply(f'/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/gs_{i}_transform_gs.ply')
            mean_xyz = torch.mean(ret['gaussian'][0]._xyz, dim=0)
            ret['gaussian'][0]._xyz = ret['gaussian'][0]._xyz - mean_xyz
            ret['gaussian'][0].save_ply(f'/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/gs_{i}_centered.ply')
            ret['gaussian'][0]._xyz = ret['gaussian'][0]._xyz / blender_scale + aabb_center
            ret['gaussian'][0].save_ply(f'/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/gs_{i}_transformed.ply')
            ret['gaussian'][0]._scaling = ret['gaussian'][0]._scaling / blender_scale
            ret['gaussian'][0]._rotation = ret['gaussian'][0]._rotation / blender_scale / blender_scale
            # gaussian_list.append(ret['gaussian'][0])
            
            # Save the results
            # video = render_utils.render_video(ret['gaussian'][0])['color']
            # imageio.mimsave(os.path.join(f"/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/{i}_gs.mp4"), video, fps=30)

            # get the motion para
            # import pdb; pdb.set_trace()
            jrange = node['joint']['range']
            jtype = node['joint']['type']
            label = sem_ref['fwd'][node['name']]
            jtype_id = joint_ref['fwd'][node['joint']['type']]

            # Check if the node has an axis field
            if 'axis' in node['joint']:
                axis_d = np.array(node['joint']['axis']['direction'])
                axis_o = np.array(node['joint']['axis']['origin'])
            else:
                axis_d = None
                axis_o = None

            # construct transformed gaussian
            if node['id'] == root_id or node['parent'] == root_id: # no transform
                # ret['gaussian'][0] = Transform_GS(ret['gaussian'][0], jtype=jtype_id, jrange=jrange, axis_d=axis_d, axis_o=axis_o)
                gaussian_list_list.append(Transform_GS_list(ret['gaussian'][0], jtype=jtype_id, jrange=jrange, axis_d=axis_d, axis_o=axis_o))
            else:
                parent_id = node['parent']
                jrange_p = tree[parent_id]['joint']['range']
                jtype_p = tree[parent_id]['joint']['type']
                jtype_p_id = joint_ref['fwd'][jtype_p]
                axis_d_p = np.array(tree[parent_id]['joint']['axis']['direction']) if 'axis' in tree[parent_id]['joint'] else None
                axis_o_p = np.array(tree[parent_id]['joint']['axis']['origin']) if 'axis' in tree[parent_id]['joint'] else None
                # ret['gaussian'][0] = Transform_GS(ret['gaussian'][0], jtype=jtype_p_id, jrange=jrange_p, axis_d=axis_d_p, axis_o=axis_o_p)
                gaussian_list_list.append(Transform_GS_list(ret['gaussian'][0], jtype=jtype_p_id, jrange=jrange_p, axis_d=axis_d_p, axis_o=axis_o_p))
            # ret['gaussian'][0].save_ply(f'/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/gs_{i}_partmoved.ply')
            # gaussian_list.append(ret['gaussian'][0])

            # # video = render_utils.render_video(ret['radiance_field'][0])['color']
            # # imageio.mimsave(os.path.join(f"/mnt/slurm_home/ywchen/projects/TRELLIS/test_latent/{sha256}_rf.mp4"), video, fps=30)

            # # video = render_utils.render_video(ret['mesh'][0])['normal']
            # # imageio.mimsave(os.path.join(f"/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/{sha256}_mesh.mp4"), video, fps=30)

        # combine all gaussian, and translate the render coordinates and render
        # import pdb; pdb.set_trace()
        articulated_gs = copy.deepcopy(ret['gaussian'][0])
        anim_video = []
        for i in range(anim_count):
            articulated_gs_frame = []
            for j in range(K): 
                articulated_gs_frame.append(gaussian_list_list[j][i])
            
            articulated_gs.combine(articulated_gs_frame)
            # rotate back to the original GS coordinate
            articulated_gs.transform_gs([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            if i == 0:
                mean_xyz = torch.mean(articulated_gs._xyz, dim=0)
            articulated_gs._xyz = articulated_gs._xyz - mean_xyz

            articulated_gs._xyz = articulated_gs._xyz * 0.5 
            articulated_gs._scaling = articulated_gs._scaling * 0.25 
            articulated_gs._rotation = articulated_gs._rotation * 0.25 * 0.25 
            articulated_gs.save_ply(f'/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/gs_all_transformed_{i}.ply')
            video = render_utils.render_oneshot(articulated_gs)['color']
            anim_video.append(video[0])
        imageio.mimsave(os.path.join(f"/mnt/slurm_home/hhchen/TRELLIS/datasets/ObjaverseXL_sketchfab/results/all_gs_{i}.mp4"), anim_video, fps=30)
        
        # import pdb; pdb.set_trace()

