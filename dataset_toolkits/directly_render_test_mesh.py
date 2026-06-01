import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence
from itertools import chain
import tempfile
import glob
import trimesh
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/mnt/slurm_home/hhchen/TRELLIS_my' #'/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render_cond(file_path, output_dir, num_views):
    # output_folder = os.path.join(output_dir, 'renders_cond', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--object', file_path,
        '--views', json.dumps(views),
        '--output_folder', output_dir,
        '--resolution', '1024',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL)
    
    # if os.path.exists(os.path.join(output_folder, 'transforms.json')):
    #     return {'sha256': sha256, 'cond_rendered': True}


def _render_cond_fid_kid(file_path, output_dir, num_views):
    # output_folder = os.path.join(output_dir, 'renders_cond', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    # 写入临时文件
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmpf:
    #     json.dump(views, tmpf)
    #     tmpf_path = tmpf.name
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--object', file_path,
        '--views', json.dumps(views),  # 改为传递文件路径
        '--output_folder', output_dir,
        '--resolution', '512',
    ]

    # print(args)
    # print([type(a) for a in args])

    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL)


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--render_object_path', type=str, required=False,
                        help='Path of the object to render')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=300,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    # load the data_split_dyn_slat.json
    with open('/mnt/slurm_home/hhchen/cageplus/indexes/data_split_singapore.json', 'r') as f:
        data_split = json.load(f)
    train_objs = data_split['test']
    base_dir = '/mnt/slurm_home/hhchen/TRELLIS/pm_results(1)/epoch_199_w=0.5_pm_label_free'
    # get the obj_path from train_objs
    all_obj_path = []
    all_output_dir = []
    for obj in train_objs:
        cat, objid = obj.split('/')
        # 匹配 ?@Table@23807/0/object.ply
        pattern = os.path.join(base_dir, f'*@{cat}@{objid}', '0', 'object.ply')
        matches = glob.glob(pattern)
        # print(matches[0])
        # import pdb; pdb.set_trace()

        # expects a flat list of strings, but seperetad by ; between different list
        objs_file_string = matches[0]

        # ply to obj
        mesh = trimesh.load(objs_file_string)
        objs_file_string = objs_file_string.replace('.ply', '.obj')
        mesh.export(objs_file_string)


        # print(objs_file_string)
        all_obj_path.append(objs_file_string)

        # get matches[0]'s parent directory
        output_dir = f"{os.path.dirname(matches[0])}/rendering_results"
        # check if the output_dir exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        all_output_dir.append(output_dir)


    start = len(all_obj_path) * opt.rank // opt.world_size
    end = len(all_obj_path) * (opt.rank + 1) // opt.world_size
    all_obj_path = all_obj_path[start:end]
    all_output_dir = all_output_dir[start:end]
    records = []

                
    print(f'Processing {len(all_obj_path)} objects...')
    print(all_output_dir)
    print(all_obj_path)
    # import pdb; pdb.set_trace()

    # process objects
    func = partial(_render_cond_fid_kid, num_views=opt.num_views)
    cond_rendered = dataset_utils.foreach_pair_for_cond(all_obj_path, all_output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    # cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    # cond_rendered.to_csv(os.path.join(opt.output_dir, f'cond_rendered_{opt.rank}.csv'), index=False)
