#python ../../gaudi_spawn.py --world_size 2 \
python \
train_dreambooth_lora_sd3.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large" \
    --dataset_name="dog" \
    --instance_prompt="a photo of sks dog" \
    --validation_prompt="a photo of sks dog in a bucket" \
    --output_dir="dog_lora_sd3" \
    --mixed_precision="bf16" \
    --rank=4 \
    --resolution=1024 \
    --train_batch_size=1 \
    --learning_rate=1e-4 \
    --max_grad_norm=1 \
    --report_to="tensorboard" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --validation_epochs=50 \
    --save_validation_images \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --gaudi_config_name="Habana/stable-diffusion" \
    --bf16

#    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
