#python ./demo/run_interaction_nlp.py --save_root="../results/20250203_try_squad_v2_use_mean_q" \
#  --gpu_id=0 \
#  --model="qwen2.5-0.5b#pretrain" \
#  --dataset="custom-squad-v2-20250202-val" \
#  --data_path="../datasets/custom-squad-v2-20250202-val" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --player_path="../players/custom-squad-v2-20250202-val/players-qwen" \
#  --loss="l1" \
#  --baseline_type="pad" \
#  --background_type="ori" \
#  --cal_batch_size=1024 \
#  --verbose=1 \
#  --sparse_mode="pq" \
#  --optimizer="sgd" \
#  --lr=1e-7 \
#  --auto_lr="v1" \
#  --momentum=0.999 \
#  --niters=50000 \
#  --qcoef=0.05 \
#  --qstd="mean-vN-v0"

# 对于generation, 还得看一下 selected_dim 要不要用gt-log-odds-1000
# 20250201: 老师要求把qstd从 vN-v0 改成 mean-vN-v0

python ./demo/run_interaction_nlp.py --save_root="../results/20250203_try_squad_v2_try_auto_data_type" \
  --gpu_id=5 \
  --model="qwen2.5-14b#pretrain" \
  --dataset="custom-squad-v2-20250202-val" \
  --data_path="../datasets/custom-squad-v2-20250202-val" \
  --selected_dim="gt-log-odds" \
  --gt_type="predict" \
  --player_path="../players/custom-squad-v2-20250202-val/players-qwen" \
  --loss="l1" \
  --baseline_type="pad" \
  --background_type="ori" \
  --cal_batch_size=512 \
  --verbose=1 \
  --sparse_mode="pq" \
  --optimizer="sgd" \
  --lr=1e-7 \
  --auto_lr="v1" \
  --momentum=0.999 \
  --niters=50000 \
  --qcoef=0.05 \
  --qstd="mean-vN-v0"



