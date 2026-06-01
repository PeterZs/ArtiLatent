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

def check_which_mesh():
    pass

def _voxelize(file, sha256, output_dir):
    # mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
    # clamp vertices to the range [-0.5, 0.5]

    # load the mesh list
    sublists = opt.render_object_path.split(" ; ")
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
        if os.path.exists(os.path.join(output_dir, 'renders', sha256, 'transforms.json')):
            json_path = os.path.join(output_dir, 'renders', sha256, 'transforms.json')
            with open(json_path, 'r') as f:
                transforms = json.load(f)
                scale = transforms['scale']
                offset = transforms['offset']
            # apply the transformation to the mesh
            vertices = np.asarray(mesh_part.vertices)  # Get the vertices of the mesh as a NumPy array
            vertices = vertices * scale + np.array(offset)  # Transform the vertices
            mesh_part.vertices = o3d.utility.Vector3dVector(vertices)  # Update the mesh with the transformed vertices
            # sample surface points from the mesh
            pcd = mesh_part.sample_points_uniformly(number_of_points=100000)

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
    down_grid_vertices_all, _, index_map = pcd_grid_vertices_all.voxel_down_sample_and_trace(voxel_size=0.03,min_bound=(-1, -1, -1), max_bound=(1, 1, 1))
    representative_indices = [indices[0] for indices in index_map if len(indices) > 0]
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), np.asarray(down_grid_vertices_all.points))

    grid_labels_all = np.concatenate(grid_labels_all, axis=0)
    grid_labels_all = grid_labels_all[representative_indices]
    grid_data_all = np.concatenate([np.asarray(down_grid_vertices_all.points), grid_labels_all], axis=1)
    np.savetxt(os.path.join(output_dir, 'voxels_label', f'{sha256}.txt'), grid_data_all)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(grid_labels_all)}

# def _voxelize(file, sha256, output_dir):
#     mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
#     # clamp vertices to the range [-0.5, 0.5]
#     vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/32, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
#     vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
#     assert np.all(vertices >= 0) and np.all(vertices < 32), "Some vertices are out of bounds"
#     vertices = (vertices + 0.5) / 32 - 0.5
#     utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
#     return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--render_object_path', type=str, required=True,
                        help='Path of the object to render')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=None)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'voxels_label'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
