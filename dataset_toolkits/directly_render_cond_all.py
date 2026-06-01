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

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/mnt/slurm_home/hhchen/TRELLIS_my' #'/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def match_gs_view_angles(num_views):
    yaws = np.linspace(np.pi/2, np.pi/2 + 2 * np.pi, num_views, endpoint=False)
    pitchs = 0.25 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, num_views, endpoint=False))
    return yaws.tolist(), pitchs.tolist()

def _render_cond(file_path, output_dir, num_views):
    # output_folder = os.path.join(output_dir, 'renders_cond', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand()) # for normal render
    offset = (0, 0) #for pnsr
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    # yaws, pitchs = match_gs_view_angles(num_views) # same views with gs render
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    # radius = [2.0] * num_views
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    # fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--object', file_path,
        '--views', json.dumps(views),
        '--output_folder', output_dir,
        '--resolution', '512',
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmpf:
        json.dump(views, tmpf)
        tmpf_path = tmpf.name
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--object', file_path,
        '--views_file', tmpf_path,  # 改为传递文件路径
        '--output_folder', output_dir,
        '--resolution', '512',
    ]
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
                        help='Number of views to render') # 3000 for fid
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
    train_objs = data_split['test'] ###

    # get the obj_path from train_objs
    all_obj_path = []
    all_output_dir = []
    for obj in train_objs:
        obj_json_path = os.path.join('/mnt/slurm_home/hhchen/cageplus/data', obj, 'train32_wonormlization.json')
        obj_id = obj.split('/')[1]
        
        output_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/renders_fid_state0/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10"
        # check if the output_dir exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        all_output_dir.append(output_dir)

        with open(obj_json_path, 'r') as ff:
            partnet_mobility = json.load(ff)

            tree = partnet_mobility['diffuse_tree']
            
            # get the 'objs' in tree, and render it
            part_id = 0
            new_objs = []
            for node in tree:
                objs = node['objs']
                new_parts = []
                # change ply to the original mesh file
                for obj in objs:
                    # change 'plys/original-27.ply' to '/mnt/slurm_home/hhchen/dataset/35059/textured_objs/original-30.obj'
                    new_part = '/mnt/slurm_home/hhchen/dataset/' + obj_id + '/textured_objs/' + obj.split('/')[-1].replace('ply', 'obj')
                    new_parts.append(new_part)
                new_objs.append(new_parts)
            # import pdb; pdb.set_trace()

            # expects a flat list of strings, but seperetad by ; between different list
            objs_file_string = " ; ".join([" ".join(sublist) for sublist in new_objs])

            print(objs_file_string)
            all_obj_path.append(objs_file_string)

    start = len(all_obj_path) * opt.rank // opt.world_size
    end = len(all_obj_path) * (opt.rank + 1) // opt.world_size
    all_obj_path = all_obj_path[start:end]
    all_output_dir = all_output_dir[start:end]
    records = []

                
    print(f'Processing {len(all_obj_path)} objects...')
    print(all_output_dir)
    # import pdb; pdb.set_trace()

    # process objects
    func = partial(_render_cond_fid_kid, num_views=opt.num_views)
    cond_rendered = dataset_utils.foreach_pair_for_cond(all_obj_path, all_output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    cond_rendered.to_csv(os.path.join(opt.output_dir, f'cond_rendered_{opt.rank}.csv'), index=False)
