model=MPSGCN
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BSO.csv \
  --model_id 'dm1024_layer2_alpha0.3_dep3_emb2_time5' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 15 \
  --d_model 1024 \
  --node_num 512 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 2 \
  --propalpha 0.3 \
  --depth 3 \
  --embed_dim 2 \
  --time_window_size 5 \
  --standard type1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BSO.csv \
  --model_id 'dm1024_layer2_alpha0.3_dep3_emb2_time5' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 12 \
  --d_model 1024 \
  --node_num 512 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 2 \
  --propalpha 0.3 \
  --depth 3 \
  --embed_dim 2 \
  --time_window_size 5 \
  --standard type1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BSO.csv \
  --model_id 'dm1024_layer2_alpha0.3_dep3_emb2_time5' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 7 \
  --d_model 1024 \
  --node_num 512 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 2 \
  --propalpha 0.3 \
  --depth 3 \
  --embed_dim 2 \
  --time_window_size 5 \
  --standard type1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BSO.csv \
  --model_id 'dm1024_layer2_alpha0.3_dep3_emb2_time5' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 3 \
  --d_model 1024 \
  --node_num 512 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 2 \
  --propalpha 0.3 \
  --depth 3 \
  --embed_dim 2 \
  --time_window_size 5 \
  --standard type1
