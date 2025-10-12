model=MPSGCN
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BOHAI.csv \
  --model_id 'dm1024_layer3_alpha0.3_dep2_emb10_time6' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 15 \
  --d_model 1024 \
  --node_num 162 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 3 \
  --propalpha 0.3 \
  --depth 2 \
  --embed_dim 10 \
  --time_window_size 6 \
  --standard type1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BOHAI.csv \
  --model_id 'dm1024_layer3_alpha0.3_dep2_emb10_time6' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 12 \
  --d_model 1024 \
  --node_num 162 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 3 \
  --propalpha 0.3 \
  --depth 2 \
  --embed_dim 10 \
  --time_window_size 6 \
  --standard type1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BOHAI.csv \
  --model_id 'dm1024_layer3_alpha0.3_dep2_emb10_time6' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 7 \
  --d_model 1024 \
  --node_num 162 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 3 \
  --propalpha 0.3 \
  --depth 2 \
  --embed_dim 10 \
  --time_window_size 6 \
  --standard type1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SST/ \
  --data_path BOHAI.csv \
  --model_id 'dm1024_layer3_alpha0.3_dep2_emb10_time6' \
  --model $model \
  --data 1D \
  --batch_size 64 \
  --itr 3 \
  --train_epochs 20 \
  --patience 5 \
  --seq_len 30 \
  --pred_len 3 \
  --d_model 1024 \
  --node_num 162 \
  --conv_channel 32 \
  --skip_channel 32 \
  --e_layers 3 \
  --propalpha 0.3 \
  --depth 2 \
  --embed_dim 10 \
  --time_window_size 6 \
  --standard type1
