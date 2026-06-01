import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, **kwargs):    
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    
    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
    )
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        sha256 = metadata.loc[k, "sha256"]
        downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    if local_path.startswith('raw/github/repos/'):
                        path_parts = local_path.split('/')
                        file_name = os.path.join(*path_parts[5:])
                        zip_file = os.path.join(output_dir, *path_parts[:5])
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            file = os.path.join(tmp_dir, file_name)
                            record = func(file, sha256)
                    else:
                        file = os.path.join(output_dir, local_path)
                        record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)


def foreach_pair_for_state(all_obj_paths, all_output_dirs, all_articulation_paths, func, max_workers=8, desc='Processing objects'):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(func, obj_path, output_dir, articulation_path)
            for obj_path, output_dir, articulation_path in zip(all_obj_paths, all_output_dirs, all_articulation_paths)
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = f.result()
            if result is not None:
                results.append(result)
    return results

def foreach_pair_for_cond(all_obj_paths, all_output_dirs, func, max_workers=8, desc='Processing objects'):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(func, obj_path, output_dir)
            for obj_path, output_dir in zip(all_obj_paths, all_output_dirs)
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = f.result()
            if result is not None:
                results.append(result)
    return results

def foreach_pair_for_voxelize_me(all_output_dirs, func, max_workers=8, desc='Processing objects'):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(func, output_dir)
            for output_dir in all_output_dirs
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = f.result()
            if result is not None:
                results.append(result)
    return results

