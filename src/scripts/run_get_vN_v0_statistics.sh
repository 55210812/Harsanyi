#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-70m#pretrain" --gpu_id=1 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-160m#pretrain" --gpu_id=2 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-410m#pretrain" --gpu_id=3 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-70m-deduped#pretrain" --gpu_id=4 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-160m-deduped#pretrain" --gpu_id=5 &
#wait
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-1b#pretrain" --gpu_id=1 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-1.4b#pretrain" --gpu_id=2 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-410m-deduped#pretrain" --gpu_id=3 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-1b-deduped#pretrain" --gpu_id=4 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-1.4b-deduped#pretrain" --gpu_id=5 &
#wait
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-2.8b#pretrain" --gpu_id=1 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-6.9b#pretrain" --gpu_id=2 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-12b#pretrain" --gpu_id=3 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-6.9b-deduped#pretrain" --gpu_id=4 &
#python ./notebooks/compute_vN_v0_statistics.py --model="pythia-12b-deduped#pretrain" --gpu_id=5 &
#wait
#echo "All done"

#python ./notebooks/compute_vN_v0_statistics.py --model="qwen2.5-0.5b#pretrain" --gpu_id=2 &
#python ./notebooks/compute_vN_v0_statistics.py --model="qwen2.5-1.5b#pretrain" --gpu_id=1 &
#python ./notebooks/compute_vN_v0_statistics.py --model="qwen2.5-3b#pretrain" --gpu_id=0 &
#wait
#python ./notebooks/compute_vN_v0_statistics.py --model="qwen2.5-7b#pretrain" --gpu_id=5 &
#python ./notebooks/compute_vN_v0_statistics.py --model="qwen2.5-14b#pretrain" --gpu_id=0 &
#python ./notebooks/compute_vN_v0_statistics.py --model="qwen2.5-32b#pretrain" --gpu_id=5
#wait
#echo "All done"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="qwen2.5-0.5b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="pad"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-1.5b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="pad" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-3b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="pad" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=2 \
#  --model="qwen2.5-7b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="pad" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds-1000-zjp" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="pad" \
#  --data_type="fp32"

# debug
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds-sample=1000" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="cn-us-from-cqa-lr1e-4-epoch100" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds-sample=1000" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch60" \
#  --data_type="auto" &

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="mean" \
#  --data_type="auto"
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds-sample=1000" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="mean" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-mean-embed" \
#  --data_type="auto"




#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-from-cqa" \
#  --data_path="../datasets/custom-cn-us-from-cqa" \
#  --selected_dim="gt-log-odds-sample=1000" \
#  --player_path="../players/custom-cn-us-from-cqa/players-qwen-from-cqa" \
#  --baseline_type="learned" \
#  --baseline_path="../saved_baseline_values/custom-cn-us-from-cqa-test/qwen2.5-14b#pretrain/baseline-from-cqa/qwen14b-embedding.npy"


#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-0.5b#pretrain" \
#  --dataset="custom-IOI-v1-20250208" \
#  --data_path="../datasets/custom-IOI-v1-20250208" \
#  --player_path="../players/custom-IOI-v1-20250208/players-qwen" \
#  --baseline_type="mean" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=2 \
#  --model="qwen2.5-1.5b#pretrain" \
#  --dataset="custom-IOI-v1-20250208" \
#  --data_path="../datasets/custom-IOI-v1-20250208" \
#  --player_path="../players/custom-IOI-v1-20250208/players-qwen" \
#  --baseline_type="mean" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=3 \
#  --model="qwen2.5-3b#pretrain" \
#  --dataset="custom-IOI-v1-20250208" \
#  --data_path="../datasets/custom-IOI-v1-20250208" \
#  --player_path="../players/custom-IOI-v1-20250208/players-qwen" \
#  --baseline_type="mean" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=4 \
#  --model="qwen2.5-7b#pretrain" \
#  --dataset="custom-IOI-v1-20250208" \
#  --data_path="../datasets/custom-IOI-v1-20250208" \
#  --player_path="../players/custom-IOI-v1-20250208/players-qwen" \
#  --baseline_type="mean" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-IOI-v1-20250208" \
#  --data_path="../datasets/custom-IOI-v1-20250208" \
#  --player_path="../players/custom-IOI-v1-20250208/players-qwen" \
#  --baseline_type="mean" &




#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-manual-30" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-30" \
#  --player_path="../players/custom-cn-us-manual-30/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch60" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch60" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-7b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-5-epoch100" \
#  --data_type="auto"




#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="deepseek-r1-distill-1.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-deepseek-r1-distill-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch100" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="deepseek-r1-distill-7b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-deepseek-r1-distill-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch100" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="deepseek-r1-distill-1.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-deepseek-r1-distill-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch100-fix-bos-anneal" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="deepseek-r1-distill-1.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-deepseek-r1-distill-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch100-fix-bos" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="deepseek-r1-distill-7b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-deepseek-r1-distill-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-4-epoch100-fix-bos-anneal" \
#  --data_type="auto" &

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-3b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-5-epoch100" \
#  --data_type="auto"
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-0.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-5-epoch100" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-1.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-5-epoch100" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-7b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=2 \
#  --model="qwen2.5-3b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="qwen2.5-1.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto" &
#
#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=5 \
#  --model="qwen2.5-0.5b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto" &


#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="qwen2.5-14b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="squad-large-from-cqa-epoch85" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-3b#pretrain" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="learned" \
#  --learned_baseline_identifier="superbowl-from-cqa-lr1e-5-epoch100-warmup-annealing" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-1.5b#init" \
#  --dataset="custom-cn-us-manual-100" \
#  --label_file_name="labels_qwen2.5-1.5b#pretrain.txt" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="correct" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-1.5b#init" \
#  --dataset="custom-cn-us-manual-100" \
#  --label_file_name="labels_qwen2.5-1.5b#pretrain.txt" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="qwen2.5-1.5b#init" \
#  --dataset="custom-cn-us-manual-100" \
#  --label_file_name="labels_qwen2.5-1.5b#pretrain.txt" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-qwen-manual-phrase" \
#  --baseline_type="pad" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="pythia-1.4b#pretrain_revision=step0" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-pythia-manual-phrase" \
#  --baseline_type="unk" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=0 \
#  --model="pythia-1.4b#init" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-pythia-manual-phrase" \
#  --baseline_type="unk" \
#  --data_type="auto"

#python ./notebooks/compute_vN_v0_statistics.py \
#  --gpu_id=1 \
#  --model="LLM360-amber-7b#pretrain_revision=ckpt000" \
#  --dataset="custom-cn-us-manual-100" \
#  --selected_dim="gt-log-odds" \
#  --gt_type="predict" \
#  --data_path="../datasets/custom-cn-us-manual-100" \
#  --player_path="../players/custom-cn-us-manual-100/players-amber-manual-phrase" \
#  --baseline_type="unk" \
#  --data_type="auto"

python ./notebooks/compute_vN_v0_statistics.py \
  --gpu_id=1 \
  --model="LLM360-amber-7b#init" \
  --dataset="custom-cn-us-manual-100" \
  --selected_dim="gt-log-odds" \
  --gt_type="predict" \
  --data_path="../datasets/custom-cn-us-manual-100" \
  --player_path="../players/custom-cn-us-manual-100/players-amber-manual-phrase" \
  --baseline_type="unk" \
  --data_type="auto"