# Video-LLaMA Frequently Asked Questions (FAQ)

## General Questions

### Q: What is Video-LLaMA?
**A:** Video-LLaMA is a multi-modal large language model that can understand and discuss video and audio content through natural language conversations. It combines visual, audio, and language understanding capabilities.

### Q: What can Video-LLaMA do?
**A:** Video-LLaMA can:
- Analyze video content and answer questions
- Describe scenes, actions, and objects
- Understand audio cues and sounds
- Process static images
- Engage in multi-turn conversations about visual content
- Generate captions and summaries

### Q: What are the system requirements?
**A:** 
- GPU: CUDA-capable with 40GB+ VRAM (A100/A6000 recommended)
- RAM: 32GB+ system memory
- Storage: 50GB+ for models and checkpoints
- OS: Linux (recommended) or Windows with WSL
- Python: 3.9 or higher

## Installation & Setup

### Q: How do I install Video-LLaMA?
**A:** Follow these steps:
```bash
# 1. Clone repository
git clone <repo-url>
cd Video-LLaMA

# 2. Create environment
conda env create -f environment.yml
conda activate videollama

# 3. Install FFmpeg
sudo apt install ffmpeg  # Linux

# 4. Download checkpoints (see QUICKSTART.md)
```

### Q: Where do I get the model checkpoints?
**A:** You need three types:
1. **Language Model**: LLaMA-2 or Vicuna from HuggingFace
2. **ImageBind**: From Facebook Research GitHub
3. **Video-LLaMA weights**: Pre-trained or fine-tuned checkpoints

See QUICKSTART.md for detailed links.

### Q: Do I need all checkpoints?
**A:** Minimum required:
- Language model (LLaMA-2 or Vicuna)
- ImageBind weights
- Vision-Language branch checkpoint

Audio branch is optional if you only process video visuals.

### Q: Can I run this on CPU?
**A:** No, Video-LLaMA requires a CUDA-capable GPU. CPU inference is not supported due to the model size and computational requirements.

## Usage Questions

### Q: How do I run the demo?
**A:** 
```bash
python demo_audiovideo.py \
    --cfg-path eval_configs/video_llama_eval_withaudio.yaml \
    --model_type llama_v2 \
    --gpu-id 0
```
Then open the Gradio URL in your browser.

### Q: What video formats are supported?
**A:** Common formats supported by FFmpeg:
- MP4 (recommended)
- AVI
- MOV
- MKV
- WebM

### Q: What's the maximum video length?
**A:** There's no hard limit, but:
- Longer videos use more memory
- Processing time increases with length
- Recommended: 30 seconds to 2 minutes for best results
- The model samples frames, so very long videos are compressed

### Q: Can I process multiple videos at once?
**A:** The demo processes one video at a time. For batch processing, you'll need to modify the code or run multiple instances.

### Q: How do I improve response quality?
**A:**
- Increase beam search (3-5)
- Adjust temperature (0.7-1.0)
- Use more frames in config (n_frms: 16)
- Use the 13B model instead of 7B
- Provide more specific questions

### Q: Why are responses slow?
**A:**
- Large model size requires significant computation
- Video processing is memory-intensive
- Solutions:
  - Use 7B model instead of 13B
  - Reduce beam search to 1
  - Process shorter videos
  - Reduce number of frames (n_frms: 4)

## Troubleshooting

### Q: I get "CUDA out of memory" error
**A:** Try these solutions:
1. Use smaller model (7B instead of 13B)
2. Reduce batch size in config
3. Enable low_resource mode
4. Reduce number of frames (n_frms: 4)
5. Process shorter videos
6. Close other GPU applications

### Q: FFmpeg not found error
**A:**
```bash
# Linux
sudo apt update
sudo apt install ffmpeg

# Verify
ffmpeg -version

# Windows
# Download from ffmpeg.org and add to PATH
```

### Q: Import errors or module not found
**A:**
```bash
# Ensure environment is activated
conda activate videollama

# Reinstall dependencies
pip install -r requirement.txt

# Install package
pip install -e .
```

### Q: Checkpoint loading fails
**A:** Verify:
- File paths in config are correct
- Checkpoint exists and is not corrupted
- Model size matches (7B with 7B, 13B with 13B)
- File permissions allow reading
- Enough disk space

### Q: Gradio interface won't open
**A:**
- Check firewall settings
- Try different port: `demo.launch(server_port=7861)`
- Use share mode: `demo.launch(share=True)`
- Check if port 7860 is already in use

### Q: Video won't upload or process
**A:**
- Check video format (use MP4)
- Verify video is not corrupted
- Try re-encoding: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`
- Check file size (very large files may timeout)

## Model & Performance

### Q: What's the difference between 7B and 13B models?
**A:**
- **7B**: Faster, requires less memory (~40GB), good quality
- **13B**: Better quality, slower, requires more memory (~80GB)

For most use cases, 7B is sufficient.

### Q: LLaMA-2 vs Vicuna - which should I use?
**A:**
- **LLaMA-2**: Official Meta model, good general performance
- **Vicuna**: Fine-tuned for conversations, may be better for chat

Both work well. Choose based on availability and preference.

### Q: Can I use this for commercial purposes?
**A:** Check the licenses of:
- LLaMA-2 (Meta's license)
- Your specific Video-LLaMA checkpoint
- This codebase (BSD 3-Clause)

Generally, research and non-commercial use is permitted.

### Q: How accurate is Video-LLaMA?
**A:** Video-LLaMA is a research model with limitations:
- May hallucinate or make errors
- Better with clear, well-lit videos
- Struggles with complex scenes or long videos
- Audio understanding is basic
- Always verify critical information

### Q: Can it understand multiple languages?
**A:** 
- English: Full support
- Chinese: Supported with appropriate language models (BiLLA, Ziya)
- Other languages: Limited, depends on base language model

## Training Questions

### Q: Can I train my own model?
**A:** Yes! See README.md for training instructions. You'll need:
- 8x A100 (80GB) GPUs
- Training datasets (WebVid, LLaVA, etc.)
- Several days of training time

### Q: Can I fine-tune on my own data?
**A:** Yes, you can fine-tune on custom video-text pairs. Prepare your data in the same format as the training datasets.

### Q: How long does training take?
**A:**
- Stage 1 (Pre-training): 2-3 days on 8x A100
- Stage 2 (Fine-tuning): 1-2 days on 8x A100

### Q: Can I train on fewer GPUs?
**A:** Possible but not recommended:
- Reduce batch size proportionally
- Training will take much longer
- May affect model quality
- Minimum: 4x A100 (40GB)

## Advanced Questions

### Q: How do I integrate Video-LLaMA into my application?
**A:** You can:
1. Use the Gradio API
2. Import the model classes directly
3. Create a REST API wrapper
4. Use the command-line interface

See the demo files for examples.

### Q: Can I run this on multiple GPUs?
**A:** Yes, for training. For inference, single GPU is typically used. Modify the config for distributed training.

### Q: How do I change the prompt template?
**A:** Edit the config file:
```yaml
model:
  prompt_template: 'Your custom template: {}'
  end_sym: "###"
```

### Q: Can I use different visual encoders?
**A:** The architecture is designed for specific encoders (ViT-G/14, ImageBind). Changing encoders requires code modifications.

### Q: How do I export the model?
**A:** The model uses PyTorch. You can save/load using standard PyTorch methods, but the full pipeline is complex.

## Getting Help

### Q: Where can I get more help?
**A:**
1. Check README.md for detailed documentation
2. Review QUICKSTART.md for setup issues
3. See CONFIG_TEMPLATE.md for configuration help
4. Search GitHub issues for similar problems
5. Create a new issue with detailed information

### Q: How do I report a bug?
**A:** Create a GitHub issue with:
- Error message (full traceback)
- Your configuration file
- System information (GPU, CUDA version, etc.)
- Steps to reproduce
- Expected vs actual behavior

### Q: How can I contribute?
**A:** Contributions welcome:
- Bug fixes
- Documentation improvements
- New features
- Training on new datasets
- Performance optimizations

## Best Practices

### Q: What are the best practices for using Video-LLaMA?
**A:**
1. **Start small**: Test with short videos first
2. **Clear videos**: Use well-lit, clear footage
3. **Specific questions**: Ask precise questions for better answers
4. **Verify outputs**: Don't trust responses blindly
5. **Monitor resources**: Watch GPU memory usage
6. **Save configs**: Keep working configurations
7. **Version control**: Track which checkpoints work best

### Q: How do I get the best results?
**A:**
- Use high-quality input videos
- Ask specific, clear questions
- Adjust temperature and beam search
- Use appropriate model size for your task
- Process videos in optimal length (30s-2min)
- Enable audio only when relevant
- Experiment with different prompts

## Limitations

### Q: What are Video-LLaMA's limitations?
**A:**
- May hallucinate or generate incorrect information
- Limited temporal understanding of long sequences
- Basic audio understanding
- Requires significant computational resources
- Not real-time processing
- May struggle with complex or ambiguous scenes
- Limited to training data domains

### Q: What should I NOT use Video-LLaMA for?
**A:** Avoid using for:
- Critical decision-making without verification
- Medical diagnosis
- Legal evidence
- Safety-critical applications
- Real-time applications
- Privacy-sensitive content without proper safeguards
