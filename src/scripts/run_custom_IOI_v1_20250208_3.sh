python ./demo/run_interaction_nlp.py --save_root="../results/20250208_try_IOI_v1_fp32" \
  --gpu_id=1 \
  --model="qwen2.5-14b#pretrain" \
  --dataset="custom-IOI-v1-20250208" \
  --data_path="../datasets/custom-IOI-v1-20250208" \
  --selected_dim="gt-log-odds" \
  --gt_type="predict" \
  --player_path="../players/custom-IOI-v1-20250208/players-qwen" \
  --loss="l1" \
  --baseline_type="pad" \
  --background_type="ori" \
  --cal_batch_size=128 \
  --verbose=1 \
  --sparse_mode="pq" \
  --optimizer="sgd" \
  --lr=1e-7 \
  --auto_lr="v1" \
  --momentum=0.999 \
  --niters=50000 \
  --qcoef=0.05 \
  --qstd="mean-vN-v0" \
  --data_type="fp32"

# 对于generation, 还得看一下 selected_dim 要不要用gt-log-odds-1000
# 20250201: 老师要求把qstd从 vN-v0 改成 mean-vN-v0


