# Video-LLaMA Configuration Guide

This guide explains all configuration options for Video-LLaMA.

## Configuration Files

Video-LLaMA uses YAML configuration files located in:
- `eval_configs/` - For inference/demo
- `train_configs/` - For training

## Inference Configuration

### Basic Structure

```yaml
model:
  arch: video_llama                    # Model architecture
  model_type: pretrain_vicuna          # Base model type
  freeze_vit: True                     # Freeze visual encoder
  freeze_qformer: True                 # Freeze Q-Former
  max_txt_len: 512                     # Maximum text length
  end_sym: "###"                       # End symbol for generation
  low_resource: False                  # Enable for limited GPU memory
  
  # Checkpoint paths
  llama_model: "ckpt/llama-2-7b-chat-hf"
  imagebind_ckpt_path: "ckpt/imagebind_path/"
  ckpt: 'ckpt/pretrained_visual_branch.pth'
  ckpt_2: 'ckpt/pretrained_audio_branch.pth'
  
  # Branch configuration
  equip_audio_branch: True             # Enable audio processing
  frozen_llama_proj: False
  fusion_head_layers: 2
  max_frame_pos: 32                    # Maximum video frames
  fusion_header_type: "seqTransf"

datasets:
  webvid:
    vis_processor:
      train:
        name: "alpro_video_eval"
        n_frms: 8                      # Number of frames to sample
        image_size: 224                # Input image size

run:
  task: video_text_pretrain
```

## Key Parameters Explained

### Model Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `model_type` | Base language model | `pretrain_vicuna`, `llama_v2` | `pretrain_vicuna` |
| `max_txt_len` | Max output tokens | 128-2048 | 512 |
| `low_resource` | Memory optimization | `True`, `False` | `False` |
| `equip_audio_branch` | Enable audio | `True`, `False` | `True` |
| `max_frame_pos` | Max video frames | 8-64 | 32 |

### Visual Processing

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `n_frms` | Frames to sample | 8 (fast), 16 (quality) |
| `image_size` | Input resolution | 224 (standard), 384 (high-res) |

### Generation Parameters (Runtime)

| Parameter | Description | Range | Recommended |
|-----------|-------------|-------|-------------|
| `num_beams` | Beam search width | 1-10 | 1 (fast), 3-5 (quality) |
| `temperature` | Randomness | 0.1-2.0 | 0.7-1.0 |
| `max_new_tokens` | Output length | 50-500 | 300 |

## Training Configuration

### Stage 1: Pre-training

```yaml
model:
  arch: video_llama
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True
  num_query_token: 32
  
  llama_model: "ckpt/vicuna-7b/"
  imagebind_ckpt_path: "ckpt/imagebind_path/"
  llama_proj_model: 'ckpt/pretrained_minigpt4.pth'
  
  equip_audio_branch: False
  frozen_llama_proj: False
  frozen_video_Qformer: False
  frozen_audio_Qformer: True

datasets:
  webvid:
    data_type: video
    build_info:
      anno_dir: path/to/webvid/annotations/
      videos_dir: path/to/webvid/videos/
    vis_processor:
      train:
        name: "alpro_video_train"
        n_frms: 8
        image_size: 224
    sample_ratio: 100

run:
  task: video_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 8e-5
  warmup_lr: 1e-6
  weight_decay: 0.05
  max_epoch: 5
  batch_size_train: 32
  batch_size_eval: 32
  num_workers: 8
  warmup_steps: 2500
  iters_per_epoch: 2500
  seed: 42
  output_dir: "output/videollama_stage1_pretrain"
  amp: True
  device: "cuda"
  world_size: 1
  distributed: True
```

### Stage 2: Fine-tuning

```yaml
model:
  # Same as stage 1, but add:
  ckpt: 'path/to/stage1_checkpoint.pth'
  max_txt_len: 320
  
  # Template for Vicuna
  end_sym: "###"
  prompt_path: "prompts/alignment_image.txt"
  prompt_template: '###Human: {} ###Assistant: '
  
  # Template for LLaMA-2
  # end_sym: "</s>"
  # prompt_template: '[INST] <<SYS>>\n \n<</SYS>>\n\n{} [/INST] '

datasets:
  llava_instruct:
    data_type: images
    build_info:
      anno_dir: path/to/llava_instruct_150k.json
      videos_dir: path/to/coco/train2014/
    num_video_query_token: 32
    tokenizer_name: "ckpt/vicuna-7b/"
    model_type: "vicuna"

run:
  init_lr: 3e-5
  min_lr: 1e-5
  max_epoch: 3
  batch_size_train: 4
  iters_per_epoch: 1000
```

## Memory Optimization

### For Limited GPU Memory (< 40GB)

```yaml
model:
  low_resource: True
  
datasets:
  webvid:
    vis_processor:
      train:
        n_frms: 4              # Reduce frames
        image_size: 224        # Keep standard size

run:
  batch_size_train: 8          # Reduce batch size
  amp: True                    # Enable mixed precision
```

### For High Memory (80GB+)

```yaml
datasets:
  webvid:
    vis_processor:
      train:
        n_frms: 16             # More frames
        image_size: 384        # Higher resolution

run:
  batch_size_train: 64         # Larger batch
```

## Model Type Selection

### LLaMA-2-Chat
```yaml
model:
  model_type: pretrain_vicuna
  llama_model: "ckpt/llama-2-7b-chat-hf"
  end_sym: "</s>"
  prompt_template: '[INST] <<SYS>>\n \n<</SYS>>\n\n{} [/INST] '
```

### Vicuna
```yaml
model:
  model_type: pretrain_vicuna
  llama_model: "ckpt/vicuna-7b/"
  end_sym: "###"
  prompt_template: '###Human: {} ###Assistant: '
```

## Audio Configuration

### Enable Audio Processing
```yaml
model:
  equip_audio_branch: True
  imagebind_ckpt_path: "ckpt/imagebind_path/"
  ckpt_2: 'ckpt/pretrained_audio_branch.pth'
  frozen_audio_Qformer: False
```

### Disable Audio (Video Only)
```yaml
model:
  equip_audio_branch: False
  frozen_audio_Qformer: True
```

## Performance Tuning

### Fast Inference
```yaml
# In config
model:
  max_frame_pos: 16
datasets:
  webvid:
    vis_processor:
      train:
        n_frms: 4

# At runtime
num_beams: 1
temperature: 1.0
max_new_tokens: 150
```

### High Quality
```yaml
# In config
model:
  max_frame_pos: 32
datasets:
  webvid:
    vis_processor:
      train:
        n_frms: 16

# At runtime
num_beams: 5
temperature: 0.7
max_new_tokens: 300
```

## Troubleshooting Configurations

### Issue: Out of Memory
```yaml
# Solution 1: Reduce batch size
run:
  batch_size_train: 4

# Solution 2: Enable low resource mode
model:
  low_resource: True

# Solution 3: Reduce frames
datasets:
  webvid:
    vis_processor:
      train:
        n_frms: 4
```

### Issue: Slow Training
```yaml
# Solution 1: Increase batch size
run:
  batch_size_train: 64

# Solution 2: Reduce workers if I/O bound
run:
  num_workers: 4

# Solution 3: Enable AMP
run:
  amp: True
```

### Issue: Poor Quality
```yaml
# Solution 1: Increase frames
datasets:
  webvid:
    vis_processor:
      train:
        n_frms: 16

# Solution 2: Adjust learning rate
run:
  init_lr: 1e-5
  
# Solution 3: More epochs
run:
  max_epoch: 10
```

## Example Configurations

### Minimal (Testing)
- Model: 7B
- Frames: 4
- Batch: 4
- Memory: ~20GB

### Standard (Production)
- Model: 7B
- Frames: 8
- Batch: 16
- Memory: ~40GB

### High Quality (Research)
- Model: 13B
- Frames: 16
- Batch: 32
- Memory: ~80GB

## Configuration Checklist

Before running, verify:
- [ ] All checkpoint paths exist
- [ ] Model size matches checkpoint size
- [ ] Batch size fits in GPU memory
- [ ] Dataset paths are correct
- [ ] Prompt template matches model type
- [ ] Audio branch config matches available checkpoints
- [ ] Output directory has write permissions
