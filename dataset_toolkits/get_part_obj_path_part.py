import os
import json
import sys
import importlib
import argparse
from easydict import EasyDict as edict
import subprocess
from itertools import chain




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--model_id', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--obj_path', type=str, required=True,
                        help='Directory to save the metadata')
    opt = parser.parse_args(sys.argv[1:])
    opt = edict(vars(opt))

    # check if obj_path exists, if exists, delete it
    if os.path.exists(opt.obj_path):
        os.remove(opt.obj_path)

    json_path= os.path.join(opt.src_dir, opt.model_id, 'train32_wonormlization.json')
    with open(json_path, 'r') as f:
        partnet_mobility = json.load(f)

    tree = partnet_mobility['diffuse_tree']
    
    # get the 'objs' in tree, and save in a new txt file
    part_id = 0
    new_objs = []
    for node in tree:
        objs = node['objs']
        new_parts = []
        # change ply to the original mesh file
        for obj in objs:
            # change 'plys/original-27.ply' to '/mnt/slurm_home/hhchen/dataset/35059/textured_objs/original-30.obj'
            new_part = '/mnt/slurm_home/hhchen/dataset/' + opt.model_id + '/textured_objs/' + obj.split('/')[-1].replace('ply', 'obj')
            new_parts.append(new_part)
        new_objs.append(new_parts)
    # import pdb; pdb.set_trace()

    # expects a flat list of strings, but seperetad by ; between different list
    objs_file_string = " ; ".join([" ".join(sublist) for sublist in new_objs])
    # flat_new_objs = list(chain.from_iterable(new_objs))
    # Join the flattened list into a single string
    # objs_file_string = " ".join(flat_new_objs)

    print(objs_file_string)

    # call command using subprocess.run
    OUTPUT_dir = f"/mnt/slurm_home/hhchen/cageplus/data/Dishwasher/{opt.model_id}/global"
    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL",
        "--source", "sketchfab",
        "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 1 done')

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/download.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir, "--world_size", "160000"
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 2 download done')
    
    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/render2.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir, 
        "--render_object_path", objs_file_string
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 3 render done')

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/voxelize.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir,
        "--render_object_path", objs_file_string
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 4 voxelize done')

    cmd= [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/extract_feature.py",
        "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 5 extract_feature done')

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/encode_ss_latent.py",
        "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 6 encode_ss_latent done')

    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/encode_latent.py",
        "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    cmd = [
        "/mnt/slurm_home/hhchen/miniconda3/envs/ga/bin/python", "dataset_toolkits/build_metadata.py",
        "ObjaverseXL", "--output_dir", OUTPUT_dir
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    print('phase 7 encode_latent done')
    part_id = part_id + 1


    





