<p align="center" width="100%">
<a target="_blank"><img src="figs/video_llama_logo.jpg" alt="Video-LLaMA" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>



# Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding

This repository contains the implementation of Video-LLaMA, a multi-modal large language model that enables natural language conversations about video and audio content.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2306.02858'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='#quick-start'><img src='https://img.shields.io/badge/Quick-Start-green'></a>
<a href='#getting-started'><img src='https://img.shields.io/badge/Setup-Guide-blue'></a>
</div>

## Features

- 🎥 **Video Understanding**: Analyze and discuss video content through natural language
- 🔊 **Audio Processing**: Understand audio cues and sound effects in videos
- 🖼️ **Image Analysis**: Process static images with detailed descriptions
- 💬 **Interactive Chat**: Engage in multi-turn conversations about visual content
- 🎯 **Instruction Following**: Respond to specific queries about video/image content
- 🌐 **Multi-lingual Support**: Support for English and Chinese (with appropriate language models)
- ⚡ **Flexible Deployment**: Run locally with Gradio interface or integrate into applications

## Capabilities

Video-LLaMA can:
- Describe scenes, actions, and objects in videos
- Identify people, places, and events
- Understand temporal sequences and relationships
- Recognize audio cues and background sounds
- Answer questions about video content
- Generate detailed captions and summaries
- Perform visual reasoning tasks

## News
- [11.14] ⭐️ The current README file is for **Video-LLaMA-2** (LLaMA-2-Chat as language decoder) only, instructions for using the previous version of Video-LLaMA (Vicuna as language decoder) can be found at [here](./README_Vicuna.md).
- [08.03] 🚀🚀 Release **Video-LLaMA-2** with [Llama-2-7B/13B-Chat](https://huggingface.co/meta-llama) as language decoder
- [06.14]  **NOTE**: The current online interactive demo is primarily for English chatting and it may **NOT** be a good option to ask Chinese questions since Vicuna/LLaMA does not represent Chinese texts very well. 
- [06.13]  **NOTE**: The audio support is **ONLY** for Vicuna-7B by now although we have several VL checkpoints available for other decoders.
- [06.08] 🚀🚀 Release the checkpoints of the audio-supported Video-LLaMA. Documentation and example outputs are also updated.    
- [05.22] ⭐️ Release **Video-LLaMA v2** built with Vicuna-7B
- [05.18] 🚀🚀 Support video-grounded chat in Chinese 
- [05.07] Release the initial version of **Video-LLaMA**, including its pre-trained and instruction-tuned checkpoints.

<p align="center" width="100%">
<a target="_blank"><img src="figs/architecture_v2.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Quick Start

### Minimal Setup (Inference Only)

1. **Install dependencies:**
   ```bash
   conda env create -f environment.yml
   conda activate videollama
   ```

2. **Download checkpoints:**
   - Language model (LLaMA-2 or Vicuna)
   - ImageBind weights
   - Video-LLaMA pre-trained weights

3. **Configure paths** in `eval_configs/video_llama_eval_withaudio.yaml`

4. **Run demo:**
   ```bash
   python demo_audiovideo.py --cfg-path eval_configs/video_llama_eval_withaudio.yaml --model_type llama_v2 --gpu-id 0
   ```

5. **Open browser** and navigate to the provided Gradio URL

## Introduction

Video-LLaMA is built on top of [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). It is composed of two core components: (1) Vision-Language (VL) Branch and (2) Audio-Language (AL) Branch.
  - **VL Branch** (Visual encoder: ViT-G/14 + BLIP-2 Q-Former)
    - A two-layer video Q-Former and a frame embedding layer (applied to the embeddings of each frame) are introduced to compute video representations. 
    - We train VL Branch on the Webvid-2M video caption dataset with a video-to-text generation task. We also add image-text pairs (~595K image captions from [LLaVA](https://github.com/haotian-liu/LLaVA)) into the pre-training dataset to enhance the understanding of static visual concepts.
    - After pre-training, we further fine-tune our VL Branch using the instruction-tuning data from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) and [VideoChat](https://github.com/OpenGVLab/Ask-Anything). 
  - **AL Branch** (Audio encoder: ImageBind-Huge) 
    - A two-layer audio Q-Former and an audio segment embedding layer (applied to the embedding of each audio segment) are introduced to compute audio representations.
    - As the used audio encoder (i.e., ImageBind) is already aligned across multiple modalities, we train AL Branch on video/image instruction data only, just to connect the output of ImageBind to the language decoder.    
- Only the Video/Audio Q-Former, positional embedding layers, and linear layers are trainable during cross-modal training.



## Example Outputs


See the `examples/` folder for sample videos and images to test the model.



## Pre-trained & Fine-tuned Checkpoints

The checkpoints contain full weights (visual encoder + audio encoder + Q-Formers + language decoder) to launch Video-LLaMA.

Pre-trained and fine-tuned model checkpoints are available for both 7B and 13B versions. Models are trained on:
- Pre-training: WebVid (2.5M video-caption pairs) and LLaVA-CC3M (595k image-caption pairs)
- Fine-tuning: Instruction-tuning data from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) and [VideoChat](https://github.com/OpenGVLab/Ask-Anything)


## Getting Started

### System Requirements

- CUDA-capable GPU (recommended: A100 40G/80G or A6000)
- 40GB+ GPU memory for inference
- 80GB+ GPU memory for training
- Linux/Windows with CUDA support
- Python 3.9+

### Installation

#### 1. Environment Setup

First, install ffmpeg (required for video processing):

**Linux:**
```bash
apt update
apt install ffmpeg
```

**Windows:**
Download and install ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.

#### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate videollama
```

Alternatively, install dependencies manually:
```bash
conda create -n videollama python=3.9
conda activate videollama
pip install -r requirement.txt
```

#### 3. Install the Package

```bash
pip install -e .
```

### Model Checkpoints

You need to download the following checkpoints before running Video-LLaMA:

#### Required Checkpoints:

1. **Language Model (LLaMA/Vicuna)**
   - Download LLaMA-2-7B-Chat or LLaMA-2-13B-Chat from [HuggingFace](https://huggingface.co/meta-llama)
   - Or download Vicuna weights (7B/13B) from [lmsys](https://huggingface.co/lmsys)
   - Place in: `ckpt/llama-2-7b-chat-hf/` or `ckpt/vicuna-7b/`

2. **Visual Encoder (ImageBind)**
   - Download from [ImageBind GitHub](https://github.com/facebookresearch/ImageBind)
   - Place in: `ckpt/imagebind_path/`

3. **Video-LLaMA Weights**
   - Download pre-trained or fine-tuned checkpoints for your chosen model size (7B/13B)
   - Vision-Language branch checkpoint
   - Audio-Language branch checkpoint (if using audio features)

#### Checkpoint Directory Structure:
```
ckpt/
├── llama-2-7b-chat-hf/          # or vicuna-7b/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
├── imagebind_path/
│   └── imagebind_huge.pth
├── pretrained_visual_branch.pth
└── pretrained_audio_branch.pth   # optional, for audio support
```

## Running the Demo

### Configuration

Before running, configure the checkpoint paths in the appropriate config file:

**For video + audio demo:** Edit `eval_configs/video_llama_eval_withaudio.yaml`
**For video-only demo:** Edit `eval_configs/video_llama_eval_only_vl.yaml`

Update the following paths:
```yaml
model:
  llama_model: "ckpt/llama-2-7b-chat-hf"  # Path to your language model
  imagebind_ckpt_path: "ckpt/imagebind_path/"  # Path to ImageBind
  ckpt: 'ckpt/pretrained_visual_branch.pth'  # Vision-Language branch weights
  ckpt_2: 'ckpt/pretrained_audio_branch.pth'  # Audio-Language branch weights (if using audio)
```

### Launch Demo

#### Video + Audio Demo (Full Features)
```bash
python demo_audiovideo.py \
    --cfg-path eval_configs/video_llama_eval_withaudio.yaml \
    --model_type llama_v2 \
    --gpu-id 0
```

For Vicuna models, use `--model_type vicuna`

#### Video-Only Demo (No Audio)
```bash
python demo_video.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --gpu-id 0
```

The demo will launch a Gradio web interface. Open the provided URL in your browser to interact with Video-LLaMA.

### Using the Demo

1. **Upload Media**: Click to upload a video or image
2. **Start Chat**: Click "Upload & Start Chat" button
3. **Ask Questions**: Type your question about the video/image content
4. **Adjust Parameters**: 
   - Beam search: Controls generation diversity (1-10)
   - Temperature: Controls randomness (0.1-2.0)
   - Audio: Enable/disable audio processing

### Example Queries

- "What is happening in this video?"
- "Describe the scene in detail"
- "What sounds can you hear?"
- "Who is in the video?"
- "What is the main action taking place?"

## Training (Advanced)

Training Video-LLaMA consists of two stages for each branch (Vision-Language and Audio-Language):

### Stage 1: Pre-training

#### Data Preparation

1. **WebVid Dataset** (2.5M video-caption pairs)
   - Download from [WebVid GitHub](https://github.com/m-bain/webvid)
   - Structure:
   ```
   webvid_train_data/
   ├── filter_annotation/
   │   └── 0.tsv
   └── videos/
       └── 000001_000050/
           └── 1066674784.mp4
   ```

2. **LLaVA-CC3M Dataset** (595K image-caption pairs)
   - Download from [LLaVA](https://github.com/haotian-liu/LLaVA)
   - Structure:
   ```
   cc3m/
   ├── filter_cap.json
   └── image/
       └── GCC_train_000000000.jpg
   ```

#### Training Commands

**Vision-Language Branch:**
```bash
# Configure paths in train_configs/visionbranch_stage1_pretrain.yaml
torchrun --nproc_per_node=8 train.py \
    --cfg-path ./train_configs/visionbranch_stage1_pretrain.yaml
```

**Audio-Language Branch:**
```bash
# Configure paths in train_configs/audiobranch_stage1_pretrain.yaml
torchrun --nproc_per_node=8 train.py \
    --cfg-path ./train_configs/audiobranch_stage1_pretrain.yaml
```

### Stage 2: Instruction Fine-tuning

#### Data Preparation

Download the following instruction-tuning datasets:
- **LLaVA-Instruct-150K**: [Download](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json)
- **MiniGPT-4 Instructions**: [Instructions](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_2_STAGE.md)
- **VideoChat Instructions**: [Download](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data)

#### Training Commands

**Vision-Language Branch:**
```bash
# Configure paths in train_configs/visionbranch_stage2_finetune.yaml
torchrun --nproc_per_node=8 train.py \
    --cfg-path ./train_configs/visionbranch_stage2_finetune.yaml
```

**Audio-Language Branch:**
```bash
# Configure paths in train_configs/audiobranch_stage2_finetune.yaml
torchrun --nproc_per_node=8 train.py \
    --cfg-path ./train_configs/audiobranch_stage2_finetune.yaml
```

### Training Configuration

Key parameters to adjust in config files:
- `llama_model`: Path to base language model
- `imagebind_ckpt_path`: Path to ImageBind weights
- `ckpt`: Path to pre-trained checkpoint (for stage 2)
- `batch_size_train`: Adjust based on GPU memory
- `max_epoch`: Number of training epochs
- `init_lr`: Initial learning rate

## Hardware Requirements

| Task | Recommended GPU | Memory | Notes |
|------|----------------|---------|-------|
| Inference (7B) | 1x A100 (40G) or A6000 | 40GB+ | Single GPU sufficient |
| Inference (13B) | 1x A100 (80G) | 80GB+ | Requires high memory |
| Pre-training | 8x A100 (80G) | 640GB total | Distributed training |
| Fine-tuning | 8x A100 (80G) | 640GB total | Distributed training |

### Memory Optimization Tips

- Use `low_resource: True` in config for limited GPU memory
- Reduce `batch_size_train` if encountering OOM errors
- Use gradient checkpointing for training
- Consider using 8-bit quantization for inference

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory (OOM)**
```
Solution:
- Reduce batch size in config file
- Use smaller model (7B instead of 13B)
- Enable low_resource mode: set low_resource: True in config
- Reduce number of video frames: decrease n_frms in config
```

**2. FFmpeg Not Found**
```
Error: ffmpeg: command not found
Solution:
- Linux: sudo apt install ffmpeg
- Windows: Download from ffmpeg.org and add to PATH
- Verify: ffmpeg -version
```

**3. Checkpoint Loading Errors**
```
Error: FileNotFoundError or checkpoint mismatch
Solution:
- Verify all checkpoint paths in config file
- Ensure checkpoint versions match (7B with 7B, 13B with 13B)
- Check file permissions
- Re-download corrupted checkpoints
```

**4. Gradio Connection Issues**
```
Error: Cannot connect to Gradio interface
Solution:
- Check firewall settings
- Try: demo.launch(share=True) for public URL
- Use different port: demo.launch(server_port=7861)
```

**5. Import Errors**
```
Error: ModuleNotFoundError
Solution:
- Ensure conda environment is activated: conda activate videollama
- Reinstall dependencies: pip install -r requirement.txt
- Install package: pip install -e .
```

**6. Video Processing Errors**
```
Error: Cannot decode video
Solution:
- Ensure video format is supported (mp4, avi, mov)
- Check video is not corrupted
- Try re-encoding: ffmpeg -i input.mp4 -c:v libx264 output.mp4
```

### Performance Tips

- **Faster Inference**: Use beam_search=1 and lower temperature
- **Better Quality**: Increase beam_search (3-5) and adjust temperature (0.7-1.0)
- **Memory Saving**: Process shorter videos or reduce frame count
- **Multi-GPU**: Modify config for distributed inference

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review configuration files for typos
3. Verify all paths are correct
4. Ensure GPU drivers and CUDA are properly installed
5. Check system meets minimum requirements

## Acknowledgement
We are grateful for the following awesome projects our Video-LLaMA arising from:
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): Enhancing Vision-language Understanding with Advanced Large Language Models
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots
* [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2): Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models 
* [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP): Improved Training Techniques for CLIP at Scale
* [ImageBind](https://github.com/facebookresearch/ImageBind): One Embedding Space To Bind Them All
* [LLaMA](https://github.com/facebookresearch/llama): Open and Efficient Foundation Language Models
* [VideoChat](https://github.com/OpenGVLab/Ask-Anything): Chat-Centric Video Understanding
* [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
* [WebVid](https://github.com/m-bain/webvid): A Large-scale Video-Text dataset
* [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main): Modularization Empowers Large Language Models with Multimodality

The logo of Video-LLaMA is generated by [Midjourney](https://www.midjourney.com/).


## Term of Use
Our Video-LLaMA is just a research preview intended for non-commercial use only. You must **NOT** use our Video-LLaMA for any illegal, harmful, violent, racist, or sexual purposes. You are strictly prohibited from engaging in any activity that will potentially violate these guidelines. 

## Citation
If you find this project useful, please cite the paper:
```
@article{videollama2023,
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  year = 2023,
  journal = {arXiv preprint arXiv:2306.02858},
  url = {https://arxiv.org/abs/2306.02858}
}
```

