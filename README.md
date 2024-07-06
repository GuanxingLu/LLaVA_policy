# LLaVA Policy

## Install

Please follow the official LLaVA repo to install the dependences: https://github.com/haotian-liu/LLaVA

## Usage

1. Convert the collected data to llava's format

```
cd LLaVA
conda activate <your_env>
python scripts/convert_homerobot_to_llava.py
```

e.g.,
```
save_sample['conversations'] = [
    {'from': 'human', 'value': '<image>\nSpecify the contact point and gripper direction of manipulating the object.',},
    {'from': 'gpt', 'value': f'The action is {action}',}
]
```

2. Train with lora
```
./scripts/v1_5/finetune_manip_lora.sh 0,1,2,3
```

3. Merge the lora checkpoints with the base

```
python scripts/merge_lora_weights.py \
    --model-path "./checkpoints/llava-v1.5-7b-manip-lora" \
    --model-base "/mnt/disk_1/yiqin/ckpt/llava-v1.5-7b" \
    --save-model-path "./checkpoints/llava-v1.5-7b-manip-lora-merge"
```

4. Test the full checkpoint in ovmm
```
HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=7 python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml --baseline_config_path projects/habitat_ovmm/configs/agent/llava_agent.yaml --data_dir data/datasets/llava_agent habitat.task.place_init=True habitat.dataset.split="train" habitat.environment.max_episode_steps=200
```

