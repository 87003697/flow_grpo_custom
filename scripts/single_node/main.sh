# 1 GPU - 通过gradient accumulation保持相同有效batch size，减少内存使用
# 策略：减少单次batch size，增加gradient_accumulation_steps保持相同的有效batch size
# 有效batch size = train_batch_size * gradient_accumulation_steps = 6 * 6 = 36 (与原来的12*3=36相同)
# 降低分辨率到256进一步减少内存使用
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3.py --config config/dgx.py:pickscore_sd3 --config.sample.num_image_per_prompt=6 --config.sample.train_batch_size=6 --config.train.batch_size=6 --config.train.gradient_accumulation_steps=6 --config.resolution=256

# # 8 GPU - 原始配置
# export NCCL_P2P_Disable=1
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29503 scripts/train_sd3.py --config config/dgx.py:pickscore_sd3 --config.sample.num_image_per_prompt=6 --config.sample.train_batch_size=6 --config.train.batch_size=6 --config.train.gradient_accumulation_steps=6 --config.resolution=256
