# 确保日志文件存在并可写
LOG_FILE="logs.out.cond_render_logs"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# render for all models with conditions
python dataset_toolkits/directly_render_cond_all.py PartMove --world_size 4
python dataset_toolkits/run_paralle_directly_render_cond_all.py

# render for all models with different states
python dataset_toolkits/directly_render_state_all.py PartMove --world_size 4
# python dataset_toolkits/directly_render_state_all_copy.py PartMove --world_size 4
python dataset_toolkits/run_paralle_directly_render_state_all.py

# get voxels
python dataset_toolkits/run_parallel_directly_voxelize_me_all.py

# get different states' voxels
python dataset_toolkits/voxelize_state.py

# get dino features from all state images, for each voxel at the static state
python dataset_toolkits/run_paralle_directly_extract_feature_all.py
python dataset_toolkits/directly_render_state_all_copy.py PartMove


