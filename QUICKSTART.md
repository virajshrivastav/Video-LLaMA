# Video-LLaMA Quick Start Guide

This guide will help you get Video-LLaMA running in the shortest time possible.

## Prerequisites

- CUDA-capable GPU (40GB+ VRAM recommended)
- Python 3.9+
- Conda or pip
- FFmpeg

## Step-by-Step Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Video-LLaMA
```

### 2. Install Dependencies
```bash
# Create conda environment
conda env create -f environment.yml
conda activate videollama

# Or use pip
pip install -r requirement.txt
```

### 3. Install FFmpeg

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to system PATH

**Verify:**
```bash
ffmpeg -version
```

### 4. Download Model Checkpoints

You need three types of checkpoints:

#### A. Language Model (Choose one)
- **LLaMA-2-7B-Chat**: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- **LLaMA-2-13B-Chat**: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
- **Vicuna-7B**: https://huggingface.co/lmsys/vicuna-7b-v1.5
- **Vicuna-13B**: https://huggingface.co/lmsys/vicuna-13b-v1.5

Place in: `ckpt/llama-2-7b-chat-hf/` or `ckpt/vicuna-7b/`

#### B. ImageBind (Audio/Visual Encoder)
- Download: https://github.com/facebookresearch/ImageBind
- Place in: `ckpt/imagebind_path/imagebind_huge.pth`

#### C. Video-LLaMA Weights
- Download pre-trained or fine-tuned weights for your model size
- Vision-Language branch: `ckpt/pretrained_visual_branch.pth`
- Audio-Language branch: `ckpt/pretrained_audio_branch.pth` (optional)

### 5. Configure Paths

Edit `eval_configs/video_llama_eval_withaudio.yaml`:

```yaml
model:
  llama_model: "ckpt/llama-2-7b-chat-hf"
  imagebind_ckpt_path: "ckpt/imagebind_path/"
  ckpt: 'ckpt/pretrained_visual_branch.pth'
  ckpt_2: 'ckpt/pretrained_audio_branch.pth'
```

### 6. Run Demo

**With Audio Support:**
```bash
python demo_audiovideo.py \
    --cfg-path eval_configs/video_llama_eval_withaudio.yaml \
    --model_type llama_v2 \
    --gpu-id 0
```

**Video Only (No Audio):**
```bash
python demo_video.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --gpu-id 0
```

**For Vicuna models:**
```bash
python demo_audiovideo.py \
    --cfg-path eval_configs/video_llama_eval_withaudio.yaml \
    --model_type vicuna \
    --gpu-id 0
```

### 7. Access Interface

Open your browser and navigate to the URL shown in terminal (typically `http://localhost:7860`)

## Using the Demo

1. **Upload**: Click to upload a video or image
2. **Start**: Click "Upload & Start Chat"
3. **Ask**: Type your question about the content
4. **Adjust**: Modify beam search and temperature for different results

## Example Questions

- "What is happening in this video?"
- "Describe the scene"
- "What sounds do you hear?"
- "Who is in the video?"
- "What is the main action?"

## Common Issues

### Out of Memory
```bash
# Use smaller model or reduce batch size
# Edit config: batch_size_train: 16 → 8
```

### FFmpeg Not Found
```bash
# Verify installation
ffmpeg -version

# Linux: sudo apt install ffmpeg
# Windows: Add to PATH
```

### Checkpoint Errors
```bash
# Verify paths in config file
# Ensure model sizes match (7B with 7B)
# Check file permissions
```

### Import Errors
```bash
# Activate environment
conda activate videollama

# Reinstall dependencies
pip install -r requirement.txt
```

## Directory Structure

```
Video-LLaMA/
├── ckpt/
│   ├── llama-2-7b-chat-hf/
│   ├── imagebind_path/
│   ├── pretrained_visual_branch.pth
│   └── pretrained_audio_branch.pth
├── eval_configs/
│   ├── video_llama_eval_withaudio.yaml
│   └── video_llama_eval_only_vl.yaml
├── demo_audiovideo.py
├── demo_video.py
└── examples/
```

## Next Steps

- Try different videos and images
- Adjust parameters for better results
- Explore training your own models (see README.md)
- Integrate into your applications

## Resources

- Full Documentation: See README.md
- Paper: https://arxiv.org/abs/2306.02858
- Issues: Check GitHub issues for solutions

## Tips

- Start with shorter videos for faster processing
- Use beam_search=1 for faster inference
- Increase temperature (0.7-1.0) for more creative responses
- Enable audio only if your video has meaningful sound
