# using clip to encode the semantic of each part of the object

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
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def _voxelize(output_dir):
    sha256='000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10'

    # load the mesh list
    model_id = output_dir.rstrip('/').split('/')[-1]
    print(model_id)  # Output: 11622
    json_path= os.path.join(output_dir, 'train32_wonormlization.json')
    with open(json_path, 'r') as f:
        partnet_mobility = json.load(f)
    tree = partnet_mobility['diffuse_tree']
    # get the 'objs' in tree
    new_objs = []
    for node in tree:
        objs = node['objs']
        new_parts = []
        # change ply to the original mesh file
        for obj in objs:
            # change 'plys/original-27.ply' to '/mnt/slurm_home/hhchen/dataset/35059/textured_objs/original-30.obj'
            new_part = '/mnt/slurm_home/hhchen/dataset/' + model_id + '/textured_objs/' + obj.split('/')[-1].replace('ply', 'obj')
            new_parts.append(new_part)
        new_objs.append(new_parts)

    # expects a flat list of strings, but seperetad by ; between different list
    render_object_path = " ; ".join([" ".join(sublist) for sublist in new_objs])
    
    sublists = render_object_path.split(" ; ")
    part_paths = [sublist.split(" ") for sublist in sublists]
    # new a numpy array to store the mesh vertices
    grid_vertices_all = []
    grid_labels_all = []
    for i, obj_path in enumerate(part_paths):
        # load the meshes for each part
        mesh_part = o3d.geometry.TriangleMesh()
        for path in obj_path:
            sub_mesh_part = o3d.io.read_triangle_mesh(path)
            # merge all meshes
            mesh_part += sub_mesh_part       
        # Swap y and z to convert Open3D -> Blender
        vertices = np.asarray(mesh_part.vertices)
        vertices_blender = vertices[:, [0, 2, 1]]
        vertices_blender = vertices_blender * np.array([1, -1, 1])
        mesh_part.vertices = o3d.utility.Vector3dVector(vertices_blender)

        # load the transformation coefficients
        os.makedirs(os.path.join(output_dir, 'global', 'voxels_1000000'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'global', 'voxels_label_1000000'), exist_ok=True)
        if os.path.exists(os.path.join(output_dir, 'global', 'renders', sha256, 'transforms.json')):
            json_path = os.path.join(output_dir, 'global', 'renders', sha256, 'transforms.json')
            with open(json_path, 'r') as f:
                transforms = json.load(f)
                scale = transforms['scale']
                offset = transforms['offset']
            # apply the transformation to the mesh
            vertices = np.asarray(mesh_part.vertices)  # Get the vertices of the mesh as a NumPy array
            vertices = vertices * scale + np.array(offset)  # Transform the vertices
            mesh_part.vertices = o3d.utility.Vector3dVector(vertices)  # Update the mesh with the transformed vertices
            # sample surface points from the mesh
            pcd = mesh_part.sample_points_uniformly(number_of_points=1000000) # 100000

            # build a voxel grid from the pcd
            points = np.clip(np.asarray(pcd.points), -0.5 + 1e-6, 0.5 - 1e-6)
            pcd.points = o3d.utility.Vector3dVector(points)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
            grid_vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            assert np.all(grid_vertices >= 0) and np.all(grid_vertices < 64), "Some vertices are out of bounds"
            grid_vertices = (grid_vertices + 0.5) / 64 - 0.5
            grid_vertices_all.append(grid_vertices)
            grid_labels = np.zeros((grid_vertices.shape[0], 1), dtype=np.float64) + i
            grid_labels_all.append(grid_labels)
        else:
            print(f"transforms.json not found for {sha256}, skipping transformation")
            continue
    # concatenate the grid vertices from all parts
    grid_vertices_all = np.concatenate(grid_vertices_all, axis=0)

    # further dowsample the grid vertices
    pcd_grid_vertices_all = o3d.geometry.PointCloud()
    pcd_grid_vertices_all.points = o3d.utility.Vector3dVector(grid_vertices_all)
    # 0.03-0.02
    down_grid_vertices_all, _, index_map = pcd_grid_vertices_all.voxel_down_sample_and_trace(voxel_size=0.01,min_bound=(-1, -1, -1), max_bound=(1, 1, 1))
    representative_indices = [indices[0] for indices in index_map if len(indices) > 0]
    utils3d.io.write_ply(os.path.join(output_dir, 'global', 'voxels_1000000', f'{sha256}.ply'), np.asarray(down_grid_vertices_all.points))

    grid_labels_all = np.concatenate(grid_labels_all, axis=0)
    grid_labels_all = grid_labels_all[representative_indices]
    grid_data_all = np.concatenate([np.asarray(down_grid_vertices_all.points), grid_labels_all], axis=1)
    np.savetxt(os.path.join(output_dir, 'global', 'voxels_label_1000000', f'{sha256}.txt'), grid_data_all)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(grid_labels_all)}

def _clip_encode(output_dir):
    sha256='000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10'

    # load the mesh list
    model_id = output_dir.rstrip('/').split('/')[-1]
    print(model_id)  # Output: 11622
    json_path= os.path.join(output_dir, 'train32_wonormlization.json')
    with open(json_path, 'r') as f:
        partnet_mobility = json.load(f)
    tree = partnet_mobility['diffuse_tree']
    meta = partnet_mobility['meta']
    obj_cat = meta['obj_cat']

    # load the clip model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # get the 'objs' in tree
    for i, node in enumerate(tree):
        part_name= node['name']
        part_text = f"a {part_name} of a {obj_cat}"
        text = clip.tokenize(part_text).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        # save the text_features
        os.makedirs(os.path.join(output_dir, 'global', 'part_text_features'), exist_ok=True)
        np.save(os.path.join(output_dir, 'global', 'part_text_features', f'{i}.npy'), text_features.cpu().numpy())

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=None)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # get file list
    # load the data_split_dyn_slat.json
    with open('/mnt/slurm_home/hhchen/cageplus/indexes/data_split_dyn_slat_all_0507.json', 'r') as f:
        data_split = json.load(f)
    train_objs = data_split['train']

    all_output_dir = []
    for obj in train_objs:
        output_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/"
        all_output_dir.append(output_dir)

    start = len(all_output_dir) * opt.rank // opt.world_size
    end = len(all_output_dir) * (opt.rank + 1) // opt.world_size
    all_output_dir = all_output_dir[start:end]
    records = []

    # filter out objects that are already processed
    # pass
    print(all_output_dir)            
    print(f'Processing {len(all_output_dir)} objects...')

    # process objects
    func = partial(_clip_encode)
    cliped = dataset_utils.foreach_pair_for_voxelize_me(all_output_dir, func, max_workers=opt.max_workers, desc='clip encoding')
    cliped = pd.concat([cliped, pd.DataFrame.from_records(records)])
    cliped.to_csv(os.path.join(opt.output_dir, f'cliped_{opt.rank}.csv'), index=False)
