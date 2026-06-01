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
from utils import sphere_hammersley_sequence, sem_ref, joint_ref
from itertools import chain


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/mnt/slurm_home/hhchen/TRELLIS_my' #'/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def load_articulation(articulation_path):
    import pdb; pdb.set_trace()
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

def _render_cond(file_path, output_dir, articulation_path, num_views, num_states):
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
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_state.py'),
        '--',
        '--object', file_path,
        '--views', json.dumps(views),
        '--num_states', str(num_states),
        '--output_folder', output_dir,
        '--articulation_path', articulation_path,
        '--resolution', '512',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL)
    
    # if os.path.exists(os.path.join(output_folder, 'transforms.json')):
    #     return {'sha256': sha256, 'cond_rendered': True}


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
    parser.add_argument('--num_views', type=int, default=24,
                        help='Number of views to render')
    parser.add_argument('--num_states', type=int, default=4,
                        help='Number of states to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    # load the data_split_dyn_slat.json
    with open('/mnt/slurm_home/hhchen/cageplus/indexes/data_split_dyn_slat_all_0507.json', 'r') as f:
        data_split = json.load(f)
    train_objs = data_split['train']

    new_train_objs = {}
    new_train_objs['train'] = []

    # get the obj_path from train_objs
    all_obj_path = []
    all_output_dir = []
    all_articulation_path = []
    for obj in train_objs:
        obj_json_path = os.path.join('/mnt/slurm_home/hhchen/cageplus/data', obj, 'train_slat32_blender.json')
        obj_id = obj.split('/')[1]
        
        # add obj_path to all_obj_path
        with open(obj_json_path, 'r') as ff:
            partnet_mobility = json.load(ff)

            tree = partnet_mobility['diffuse_tree']
            new_objs = []
            for node in tree:
                objs = node['objs']
                new_parts = []
                # change ply to the original mesh file
                for part in objs:
                    # change 'plys/original-27.ply' to '/mnt/slurm_home/hhchen/dataset/35059/textured_objs/original-30.obj'
                    new_part = '/mnt/slurm_home/hhchen/dataset/' + obj_id + '/textured_objs/' + part.split('/')[-1].replace('ply', 'obj')
                    new_parts.append(new_part)
                new_objs.append(new_parts)
            # import pdb; pdb.set_trace()

            # expects a flat list of strings, but seperetad by ; between different list
            objs_file_string = " ; ".join([" ".join(sublist) for sublist in new_objs])

            # print(objs_file_string)
            all_obj_path.append(objs_file_string)

        # add output_dir to all_output_dir
        output_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/renders_state/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10"
        # check if the output_dir exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        all_output_dir.append(output_dir)

        # add articulation_path to all_articulation_path
        articulation_path = os.path.join('/mnt/slurm_home/hhchen/cageplus/data', obj, 'train_slat32_blender.json')
        all_articulation_path.append(articulation_path)

        # check if the /mnt/slurm_home/hhchen/cageplus/data/StorageFurniture/35059/global/renders_cond/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10/023.png exists
        if  os.path.exists(os.path.join(output_dir, '003_023.png')):
            # check if exists imgaes in the folder of 
            # img_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/renders_state/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10/007_047.png"
            feature_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/features_state1000000_001/dinov2_vitl14_reg/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10.npz"
            voxel_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/voxels_label_1000000/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10.txt"
            if  os.path.exists(feature_dir):
                # print(f'{obj} does exist')
                # new_train_objs['train'].append(obj)
                feats = np.load(feature_dir)
                indices =feats['indices']
                voxel_label = np.loadtxt(voxel_dir)[:,-1]
                if voxel_label.shape[0] != indices.shape[0]:
                    print('voxel_label.shape[0] != indices.shape[0]')
                    # new_train_objs['train'].append(obj)
                    continue
                else:
                    print(f'{obj} does exist')
                    new_train_objs['train'].append(obj)
                # import pdb; pdb.set_trace()
                continue
            else:
                # print(f'{obj}_rendering exist')
                pass
        # # check if exists folder of ply
        # ply_dir = os.path.join('/mnt/slurm_home/hhchen/cageplus/data', obj, 'plys')
        # if not os.path.exists(ply_dir):
        #     print(f'{obj} does not exist')
        #     new_train_objs['train'].append(obj)
        else:
            pass
            # new_train_objs['train'].append(obj)

    # save the new_train_objs
    print(len(new_train_objs['train']))
    with open('/mnt/slurm_home/hhchen/cageplus/indexes/data_split_dyn_slat_all_0507_debug_new.json', 'w') as f:
        json.dump(new_train_objs, f)


    # start = len(all_obj_path) * opt.rank // opt.world_size
    # end = len(all_obj_path) * (opt.rank + 1) // opt.world_size
    # all_obj_path = all_obj_path[start:end]
    # all_output_dir = all_output_dir[start:end]
    # all_articulation_path = all_articulation_path[start:end]
            
    # print(f'Processing {len(all_obj_path)} objects...')
    # print(all_output_dir)
    # # import pdb; pdb.set_trace()
    # # articulation_list = load_articulation(all_articulation_path[0])

    # # process objects
    # records = []
    # func = partial(_render_cond, num_views=opt.num_views, num_states=opt.num_states)
    # cond_rendered = dataset_utils.foreach_pair_for_state(all_obj_path, all_output_dir, all_articulation_path, func, max_workers=opt.max_workers, desc='Rendering objects')
    # cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    # cond_rendered.to_csv(os.path.join(opt.output_dir, f'state_rendered_{opt.rank}.csv'), index=False)
