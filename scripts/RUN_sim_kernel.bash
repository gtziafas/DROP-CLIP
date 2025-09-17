echo 'SIM KERNEL ABLATION, eval CLS, USING paired,0.95,scene'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,scene 
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel open,scene 
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,all 
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval RefSegm (Open+neg-scene), USING THR@0.95, paired, scene negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, open'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,scene  --sim_negatives scene
echo '-----------------------------'
echo 'Open Kernel at Scene, open'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel open,scene --sim_negatives scene
echo '-----------------------------'
echo 'Cls-only Kernel at All, open'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,all --sim_negatives scene
echo '-----------------------------'
echo 'Open Kernel at All (w/ CLS negatives), open'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel open,all --sim_negatives scene
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval SemSegm (CLS+neg-all), USING THR@0.95, paired, all negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,scene  --sim_negatives all
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel open,scene  --sim_negatives all
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,all  --sim_negatives all
echo '-----------------------------'
echo 'Open Kernel at All (w/ CLS negatives), open'
python3 run_eval.py --split val --chp_folder /home/p300488/REGRAD-Ref/data/Blender-new/checkpoints --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel open,all --sim_negatives all
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval Open, USING THR@0.95, paired, all negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,scene  --sim_negatives all
echo '-----------------------------'
echo 'Open Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel open,scene  --sim_negatives all
echo '-----------------------------'
echo 'Cls-only Kernel at All, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,all  --sim_negatives all
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval CLS, USING THR@0.95, paired, generic negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,scene  --sim_negatives generic
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel open,scene  --sim_negatives generic
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,all  --sim_negatives generic
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval Open, USING THR@0.95, paired, generic negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,scene  --sim_negatives generic
echo '-----------------------------'
echo 'Open Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel open,scene  --sim_negatives generic
echo '-----------------------------'
echo 'Cls-only Kernel at All, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,all  --sim_negatives generic
echo '-----------------------------'


echo 'SIM KERNEL ABLATION, eval CLS, USING THR@0.95, paired, no negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,scene  --sim_negatives ''
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel open,scene  --sim_negatives ''
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario cls --use_sim_kernel cls,all  --sim_negatives ''
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval Open, USING THR@0.95, paired, no negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, Open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,scene  --sim_negatives ''
echo '-----------------------------'
echo 'Open Kernel at Scene, Open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel open,scene  --sim_negatives ''
echo '-----------------------------'
echo 'Cls-only Kernel at All, Open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.95 --eval_scenario open --use_sim_kernel cls,all  --sim_negatives ''
echo '-----------------------------'


echo 'SIM KERNEL ABLATION, eval CLS, USING argmax scene negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel cls,scene  --sim_negatives scene  --sim_method argmax
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel open,scene  --sim_negatives scene --sim_method argmax
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel cls,all  --sim_negatives scene --sim_method argmax
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval Open, USING argmax, scene negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel cls,scene --sim_negatives scene  --sim_method argmax
echo '-----------------------------'
echo 'Open Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel open,scene --sim_negatives scene  --sim_method argmax
echo '-----------------------------'
echo 'Cls-only Kernel at All, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel cls,all  --sim_negatives scene  --sim_method argmax
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval CLS, USING argmax generic negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel cls,scene  --sim_negatives generic  --sim_method argmax
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel open,scene  --sim_negatives generic --sim_method argmax
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel cls,all  --sim_negatives generic --sim_method argmax
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval Open, USING argmax, generic negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel cls,scene --sim_negatives generic  --sim_method argmax
echo '-----------------------------'
echo 'Open Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel open,scene --sim_negatives generic  --sim_method argmax
echo '-----------------------------'
echo 'Cls-only Kernel at All, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel cls,all  --sim_negatives generic  --sim_method argmax
echo '-----------------------------'


echo 'SIM KERNEL ABLATION, eval CLS, USING argmax all negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel cls,scene  --sim_negatives all  --sim_method argmax
echo '-----------------------------'
echo 'Open Kernel at Scene, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel open,scene  --sim_negatives all --sim_method argmax
echo '-----------------------------'
echo 'Cls-only Kernel at All, cls'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario cls --use_sim_kernel cls,all  --sim_negatives all --sim_method argmax
echo '-----------------------------'

echo 'SIM KERNEL ABLATION, eval Open, USING argmax, all negatives'
echo '-----------------------------'
echo 'Cls-only Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel cls,scene --sim_negatives all  --sim_method argmax
echo '-----------------------------'
echo 'Open Kernel at Scene, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel open,scene --sim_negatives all  --sim_method argmax
echo '-----------------------------'
echo 'Cls-only Kernel at All, open'
python3 run_eval.py --use_obj_prior 1 --use_visibility 0 --use_similarity 1 --sim_thr 0.0 --eval_scenario open --use_sim_kernel cls,all  --sim_negatives all  --sim_method argmax
echo '-----------------------------'

