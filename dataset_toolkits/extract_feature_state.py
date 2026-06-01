import os
import copy
import sys
import json
import importlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image


torch.set_grad_enabled(False)


def get_data(frames, sha256, output_dir):
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(output_dir, 'renders_state', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    # os.makedirs(os.path.join(opt.output_dir, 'features_state', feature_name), exist_ok=True)

    # load model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    # get file list
    # if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
    #     metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    # else:
    #     raise ValueError('metadata.csv not found')
    # if opt.instances is not None:
    #     with open(opt.instances, 'r') as f:
    #         instances = f.read().splitlines()
    #     metadata = metadata[metadata['sha256'].isin(instances)]
    # else:
    #     if opt.filter_low_aesthetic_score is not None:
    #         metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
    #     if f'feature_{feature_name}' in metadata.columns:
    #         metadata = metadata[metadata[f'feature_{feature_name}'] == False]
    #     metadata = metadata[metadata['voxelized'] == True]
    #     metadata = metadata[metadata['rendered'] == True]

    # get file list by loading the data_split_dyn_slat.json
    with open('/mnt/slurm_home/hhchen/cageplus/indexes/data_split_dyn_slat_all_0507.json', 'r') as f:
        data_split = json.load(f)
    train_objs = data_split['train']
    all_output_dir = []
    for obj in train_objs:
        output_dir = f"/mnt/slurm_home/hhchen/cageplus/data/{obj}/global/"
        all_output_dir.append(output_dir)

    start = len(all_output_dir) * opt.rank // opt.world_size
    end = len(all_output_dir) * (opt.rank + 1) // opt.world_size
    all_output_dir = all_output_dir[start:end]
    print(f"Processing {len(all_output_dir)} objects")
    print(all_output_dir)
    records = []

    # filter out objects that are already processed
    # sha256s = list(metadata['sha256'].values)
    # for sha256 in copy.copy(sha256s):
    #     if os.path.exists(os.path.join(opt.output_dir, 'features_state', feature_name, f'{sha256}.npz')):
    #         records.append({'sha256': sha256, f'feature_{feature_name}' : True})
    #         sha256s.remove(sha256)

    # extract features
    sha256 = "000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10"
    num_states = 8
    num_views = 48
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            def loader(output_dir):
                try:
                    with open(os.path.join(output_dir, 'renders_state', sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    for datum in get_data(frames, sha256, output_dir):
                        datum['image'] = transform(datum['image'])
                        data.append(datum)
                    positions_list = []
                    for i in range(num_states):
                        position = np.loadtxt(os.path.join(output_dir, 'voxels_state_1000000', f'{sha256}_{i}.txt'))
                        positions_list.append(position[:, :3])
                    load_queue.put((sha256, data, positions_list, output_dir))
                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")

            loader_executor.map(loader, all_output_dir)
            
            def saver(sha256, pack, patchtokens, uv, output_dir):
                pack['patchtokens'] = F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16) #(5228, 1024)
                save_path = os.path.join(output_dir, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'feature_{feature_name}' : True})
                
            for _ in tqdm(range(len(all_output_dir)), desc="Extracting features"):
                sha256, data, positions_list, output_dir = load_queue.get()
                # import pdb; pdb.set_trace()
                # if os.path.exists(os.path.join(output_dir, 'features_state1000000/dinov2_vitl14_reg/000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10.npz')):
                #     continue
                position0 = torch.from_numpy(positions_list[0]).float().cuda()
                indices0 = ((position0 + 0.5) * 64).long()
                assert torch.all(indices0 >= 0) and torch.all(indices0 < 64), "Some vertices are out of bounds"
                n_views = len(data)
                N = position0.shape[0]
                pack = {
                    'indices': indices0.cpu().numpy().astype(np.uint8),
                }
                patchtokens_lst = []
                uv_lst = []
                for i in range(0, n_views, opt.batch_size):
                    batch_data = data[i:i+opt.batch_size]
                    bs = len(batch_data)
                    # Determine which positions state to use
                    # Each num_views views use one positions_list entry
                    positions_idx = (i // num_views)
                    positions = torch.from_numpy(positions_list[positions_idx]).float().cuda()
                    indices = ((positions + 0.5) * 64).long()
                    # assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
                    batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                    batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                    batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                    features = dinov2_model(batch_images, is_training=True)
                    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                patchtokens = torch.cat(patchtokens_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                # save features
                output_dir = os.path.join(output_dir, 'features_state1000000_001', feature_name)
                os.makedirs(output_dir, exist_ok=True)
                saver_executor.submit(saver, sha256, pack, patchtokens, uv, output_dir)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        import traceback; traceback.print_exc()
        
    # records = pd.DataFrame.from_records(records)
    # records.to_csv(os.path.join(output_dir, f'feature_{feature_name}_{opt.rank}.csv'), index=False)
        