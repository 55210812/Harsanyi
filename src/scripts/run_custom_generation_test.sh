#python ./demo/run_interaction_nlp.py --save_root="../results/20250118_try_generation_test" \
#  --gpu_id=1 \
#  --model="llama-7b#pretrain" \
#  --dataset="custom-generation-test" \
#  --data_path="../datasets/custom-generation-test" \
#  --selected_dim="gt-log-odds-sample=1000" \
#  --gt_type="predict" \
#  --player_path="../players/custom-generation-test/players-llama-manual" \
#  --loss="l1" \
#  --baseline_type="unk" \
#  --background_type="ori" \
#  --cal_batch_size=1024 \
#  --verbose=1 \
#  --sparse_mode="pq" \
#  --optimizer="sgd" \
#  --lr=1e-7 \
#  --auto_lr="v1" \
#  --momentum=0.999 \
#  --niters=50000 \
#  --qcoef=0.05


#python ./demo/run_interaction_nlp.py --save_root="../results/20250118_try_generation_test" \
#  --gpu_id=1 \
#  --model="OPT-1.3b#pretrain" \
#  --dataset="custom-generation-test" \
#  --data_path="../datasets/custom-generation-test" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --player_path="../players/custom-generation-test/players-opt-manual" \
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
#  --qcoef=0.05


python ./demo/run_interaction_nlp.py --save_root="../results/20250118_try_generation_test" \
  --gpu_id=1 \
  --model="pythia-6.9b#pretrain" \
  --dataset="custom-generation-test" \
  --data_path="../datasets/custom-generation-test" \
  --selected_dim="gt-log-odds" \
  --gt_type="predict" \
  --player_path="../players/custom-generation-test/players-pythia-manual" \
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
  --qcoef=0.05

# 对于generation, 还得看一下 selected_dim 要不要用gt-log-odds-1000