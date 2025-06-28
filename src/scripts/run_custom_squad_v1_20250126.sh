python ./demo/run_interaction_nlp.py --save_root="../results/20250126_try_squad_v1_use_mean_q" \
  --gpu_id=0 \
  --model="pythia-70m#pretrain" \
  --dataset="custom-squad-v1-20250126-val" \
  --data_path="../datasets/custom-squad-v1-20250126-val" \
  --selected_dim="gt-log-odds" \
  --gt_type="predict" \
  --player_path="../players/custom-squad-v1-20250126-val/players-pythia" \
  --loss="l1" \
  --baseline_type="unk" \
  --background_type="ori" \
  --cal_batch_size=1024 \
  --verbose=1 \
  --sparse_mode="pq" \
  --optimizer="sgd" \
  --lr=1e-7 \
  --auto_lr="v1" \
  --momentum=0.999 \
  --niters=50000 \
  --qcoef=0.05 \
  --qstd="mean-vN-v0"

# 对于generation, 还得看一下 selected_dim 要不要用gt-log-odds-1000
# 20250201: 老师要求把qstd从 vN-v0 改成 mean-vN-v0



