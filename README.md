### Quantization using Different Models.

```markdown
 Quantization of LLaMA-3.2-1B and GPT-2 Models

This repository contains the implementation and evaluation of quantization techniques applied to the LLaMA-3.2-1B and GPT-2 models to optimize memory usage, inference speed, and performance. The methods explored include AWQ, GGUF (Q4_K_M and Q8_0), and GPTQ, with detailed comparisons of memory efficiency, inference time, and response quality.

Project Overview

The goal of this project is to assess the impact of quantization on large language models, specifically LLaMA-3.2-1B and GPT-2, in terms of resource efficiency and output quality. The experiments were conducted on CUDA-enabled GPUs (T4) using Python, PyTorch, and libraries such as Hugging Face Transformers, AutoAWQ, AutoGPTQ, and llama.cpp.

Quantization Methods
- AWQ (Activation-aware Weight Quantization)**: Applied to LLaMA-3.2-1B with 4-bit precision, achieving ~60% memory reduction.
- GGUF (Q4_K_M and Q8_0)**: Converted LLaMA-3.2-1B to GGUF format and quantized to Q4_K_M (~50% size reduction) and Q8_0 (~30% size reduction).
- GPTQ: Applied to both LLaMA-3.2-1B (~45.24% memory reduction) and GPT-2 (~60.7% memory reduction) with 4-bit precision.
- LLM.int8()**: Tested on GPT-2 with mixed-precision quantization, achieving minimal accuracy loss (MSE: 1.5778e-05).

Key Findings
- Memory Efficiency: All quantization methods significantly reduced memory usage, with AWQ and GPTQ showing the largest reductions for LLaMA-3.2-1B and GPT-2, respectively.
- Inference Time: Quantized models (especially GGUF Q4_K_M) were faster than their FP16 counterparts, but GPTQ models showed slower inference times compared to originals.
- Response Quality: Quantization introduced trade-offs, with factual prompts retaining reasonable accuracy, while creative tasks (e.g., poem generation) showed degradation, particularly with GGUF Q8_0 and GPTQ.


 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/model-quantization.git
   cd model-quantization
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers autoawq autogptq bitsandbytes datasets psutil
   ```

3. For GGUF quantization, set up llama.cpp:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```

4. Ensure a CUDA-enabled GPU is available (e.g., NVIDIA T4).

## Usage

### 1. AWQ Quantization (LLaMA-3.2-1B)
Run the AWQ quantization script:
```bash
python scripts/awq_quantization.py
```
- Loads LLaMA-3.2-1B from Hugging Face.
- Applies 4-bit AWQ quantization with 256 C4 dataset samples for calibration.
- Saves the quantized model to `models/llama-awq`.

### 2. GGUF Quantization (LLaMA-3.2-1B)
Convert and quantize the model:
```bash
python scripts/gguf_conversion.py
```
- Converts LLaMA-3.2-1B to GGUF FP16 format.
- Quantizes to Q4_K_M and Q8_0 using llama.cpp.
- Outputs models to `models/llama-gguf`.

### 3. GPTQ Quantization (LLaMA-3.2-1B and GPT-2)
Run the GPTQ quantization script:
```bash
python scripts/gptq_quantization.py
```
- Quantizes LLaMA-3.2-1B or GPT-2 to 4-bit precision with 128 C4 samples.
- Saves models to `models/llama-gptq` or `models/gpt2-gptq`.

### 4. Evaluation
Evaluate memory, inference time, and response quality:
```bash
python scripts/evaluation.py
```
- Tests original and quantized models on predefined prompts.
- Outputs results to `results/`.

## Results

### LLaMA-3.2-1B
- AWQ:
  - Memory: ~1543 MB (60% reduction from ~3900 MB).
  - Inference Time: Slower than original.
  - Response Quality: Comparable for factual prompts, less focused for creative tasks.
- GGUF Q4_K_M:
  - Model Size: ~1.2 GB (50% reduction).
  - Inference Time: 15.01–35.27s (3-7x faster than FP16).
  - Response Quality: Reasonable for factual prompts, poor for creative tasks.
- GGUF Q8_0:
  - Model Size: ~1.8 GB (30% reduction).
  - Inference Time: 54.51–81.94s (faster than FP16, slower than Q4_K_M).
  - Response Quality: Lowest accuracy, often irrelevant.
- GPTQ:
  - Memory: ~5708.85 MB (45.24% reduction).
  - Inference Time: Slower than original.
  - Response Quality: Less detailed, occasional inaccuracies.

### GPT-2
- LLM.int8():
  - MSE: 1.5778e-05 (high fidelity).
  - Memory: Not explicitly measured but supports efficient operations.
- GPTQ:
  - Memory: 316.89 MB (60.7% reduction from 805.60 MB).
  - Inference Time: ~2.7737s (slower than original ~1.4172s).
  - Perplexity: 180.87 (moderate degradation).
  - Response Quality: Acceptable for factual prompts, less coherent for creative tasks.

## Prerequisites
- Python 3.11+
- CUDA-enabled GPU (e.g., NVIDIA T4)
- Libraries: `torch`, `transformers`, `autoawq`, `autogptq`, `bitsandbytes`, `datasets`, `psutil`
- llama.cpp for GGUF quantization

## Acknowledgments
- Hugging Face for providing model weights and datasets.
- llama.cpp for GGUF conversion and quantization tools.
- AutoAWQ and AutoGPTQ libraries for efficient quantization.
```
