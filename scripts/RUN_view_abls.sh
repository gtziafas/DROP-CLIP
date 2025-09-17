echo '--------------------------------------------------------------------------------'
echo 'n_views 1'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 1
echo '--------------------------------------------------------------------------------'
echo 'n_views 3'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 3
echo '--------------------------------------------------------------------------------'
echo 'n_views 6'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 6
echo '--------------------------------------------------------------------------------'
echo 'n_views 9'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 9
echo '--------------------------------------------------------------------------------'
echo 'n_views 12'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 12
echo '--------------------------------------------------------------------------------'

echo 'n_views 24'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 24
echo '--------------------------------------------------------------------------------'

echo 'n_views 36'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 36
echo '--------------------------------------------------------------------------------'

echo 'n_views 48'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 48
echo '--------------------------------------------------------------------------------'

echo 'n_views 60'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 60
echo '--------------------------------------------------------------------------------'

echo 'n_views 73'
python3 run_eval.py --use_obj_prior 1 --use_visibility 1 --use_similarity 0 --sim_thr 0.95 --sim_prompts cls --use_sim_kernel cls,scene --n_views 73


