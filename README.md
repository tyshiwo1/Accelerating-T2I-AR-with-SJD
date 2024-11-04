# Accelerating T2I-AR with SJD
Implementation of [Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding](https://arxiv.org/pdf/2410.01699)

## Performance

- Results on [Emu3](https://github.com/baaivision/Emu3) 
  <img src="assets/emu3-quali.jpg" alt="drawing" width="600"/>

- Results on [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) 
  <img src="assets/real-teaser.jpg" alt="drawing" width="600"/>

## Run

### demos

#### Lumina-mGPT

```bash
CUDA_VISIBLE_DEVICES=0 python test_lumina_mgpt.py
```

#### Emu3

```bash
CUDA_VISIBLE_DEVICES=0 python test_emu3.py
```

#### LlamaGen

```bash
CUDA_VISIBLE_DEVICES=0 python test_llamagen.py
```

## Acknowledge

Our code is based on [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT), [Emu3](https://github.com/Alpha-VLLM/Lumina-mGPT), [LlamaGen](https://github.com/FoundationVision/LlamaGen), and [Anole](https://github.com/GAIR-NLP/anole). We would like to express our gratitude to [Tianwei Xiong](https://github.com/SilentView) for his assistance.