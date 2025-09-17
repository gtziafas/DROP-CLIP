
echo 'voxel 0.002'
echo '-----------------------------'
python3 run_eval.py  --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_visibility 0 --use_similarity 1 --use_obj_prior 1 --sim_thr 0.95 --voxel_size 0.002
echo '-----------------------------'

echo 'voxel 0.004'
echo '-----------------------------'
python3 run_eval.py  --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_visibility 0 --use_similarity 1 --use_obj_prior 1 --sim_thr 0.95 --voxel_size 0.004
echo '-----------------------------'


echo 'voxel 0.006'
echo '-----------------------------'
python3 run_eval.py  --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_visibility 0 --use_similarity 1 --use_obj_prior 1 --sim_thr 0.95 --voxel_size 0.004
echo '-----------------------------'

echo 'voxel 0.008'
echo '-----------------------------'
python3 run_eval.py  --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_visibility 0 --use_similarity 1 --use_obj_prior 1 --sim_thr 0.95 --voxel_size 0.004
echo '-----------------------------'

