### Fine-Tuning Large Language Models with Quantization and Parameter-Efficient Methods
This repository contains code and documentation for fine-tuning large language models (LLMs) using various parameter-efficient fine-tuning (PEFT) techniques, including DoRA (Differentiated Optimization of Representations and Adapters), GRPO (Generalized Reward-augmented Policy Optimization), and LoRA (Low-Rank Adaptation). The models fine-tuned in this project include LLaMA-3.1-8B-Instruct (quantized) and Dolly-V2-3B, optimized for memory efficiency and performance on specific tasks.
## Overview
The project demonstrates three distinct fine-tuning approaches:

DoRA Fine-Tuning of the quantized LLaMA-3.1-8B-Instruct model on the FineTome-100k dataset.
GRPO Fine-Tuning of the quantized LLaMA-3.1-8B-Instruct model using the Anthropic HH-RLHF dataset with custom reward functions.
LoRA Fine-Tuning of the Dolly-V2-3B model on the LaMini-instruction dataset for improved instruction-following capabilities.

Each approach is implemented in a Google Colab environment, leveraging quantization (4-bit or 8-bit) and PEFT techniques to enable efficient training on limited hardware, such as a T4 or L4 GPU.
## Objectives

Fine-tune large language models to improve performance on specific conversational or instruction-following tasks.
Utilize quantization and PEFT methods (DoRA, LoRA, GRPO) to reduce memory and computational requirements.
Provide reproducible pipelines for fine-tuning, model saving, and deployment.
Share fine-tuned models and adapters via the Hugging Face Hub.

## Prerequisites
Hardware

GPU: NVIDIA T4 (for DoRA and LoRA) or L4 (for GRPO) with at least 16GB VRAM.
Environment: Google Colab or similar cloud-based platform with GPU support.

Dependencies
Install the required Python packages:
pip install torch transformers datasets peft trl bitsandbytes huggingface_hub
pip install git+https://github.com/huggingface/peft.git
pip install unsloth==2025.6.5

Key Libraries

torch: Tensor operations and GPU support.
transformers: Model loading and tokenization.
datasets: Dataset loading and preprocessing.
peft: Parameter-efficient fine-tuning (DoRA/LoRA).
trl: Supervised and reinforcement fine-tuning (SFTTrainer, GRPOTrainer).
bitsandbytes: Quantization for memory efficiency.
unsloth: Optimized model handling for GRPO.
huggingface_hub: Model uploading and sharing.

## Fine-Tuning Processes
1. DoRA Fine-Tuning (LLaMA-3.1-8B-Instruct)

Model: devatar/quantized_Llama-3.1-8B-Instruct (quantized).
Dataset: FineTome-100k (100,000 conversation samples).
Method: DoRA with LoRA configuration (rank=8, lora_alpha=16, dropout=0.05).
Training Setup:
Batch size: 1 (effective batch size of 8 with gradient accumulation).
Optimizer: Paged AdamW 8-bit.
Learning rate: 2e-5 with cosine scheduler.
Max steps: 500.


Output: Fine-tuned model and adapters saved to ./llama-3.1-8b-dorafinetuned and uploaded to Hugging Face.
Key Features:
Memory-efficient fine-tuning with bfloat16 precision and no cache.
Targets specific transformer modules (q_proj, k_proj, v_proj, o_proj).
Optimized for T4 GPU in Google Colab.



2. GRPO Fine-Tuning (LLaMA-3.1-8B-Instruct)

Model: devatar/quantized_Llama-3.1-8B-Instruct (4-bit quantized).
Dataset: Anthropic HH-RLHF (subset of 1,000 samples).
Method: GRPO with LoRA (rank=16, lora_alpha=16, dropout=0).
Reward Functions:
match_format_exactly: Rewards non-empty responses longer than 10 characters.
conversational_quality: Rewards responses closer to "chosen" than "rejected" responses (using difflib for similarity).


Training Setup:
Batch size: 1 (effective batch size of 16 with gradient accumulation).
Optimizer: AdamW 8-bit.
Learning rate: 5e-6 with cosine scheduler.
Max steps: 50.


Output: Fine-tuned model saved to ./fine_tuned_llama31_8b (merged 16-bit format).
Key Features:
4-bit quantization for reduced memory usage.
Custom reward functions for conversational alignment.
Gradient checkpointing via Unsloth for memory efficiency.



3. LoRA Fine-Tuning (Dolly-V2-3B)

Model: databricks/dolly-v2-3b (3 billion parameters, 8-bit quantized).
Dataset: LaMini-instruction (subset of 200 samples).
Method: LoRA (rank=256, lora_alpha=512, dropout=0.05).
Training Setup:
Batch size: 1.
Learning rate: 1e-4.
Epochs: 3.
Precision: float16 (FP16 enabled).


Output: Fine-tuned model with 83 million trainable parameters (2.93% of total).
Training Results:
Training loss decreased from 0.5877 to 0.1990 over 3 epochs.
Validation loss increased slightly (0.5622 to 0.6278), indicating potential overfitting.


Key Features:
Optimized for instruction-following tasks.
Suitable for deployment in chatbots or content generation tools.


Accessing Fine-Tuned Models

DoRA Model: Available at Hugging Face.
GRPO Model: Saved locally as ./fine_tuned_llama31_8b.
LoRA Model: Available at Hugging Face.

Example Inference
To perform inference with the fine-tuned models:
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "avinashhm/llama-3.1-8b-dora-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Future Improvements

Hyperparameter Tuning: Experiment with different ranks (r), lora_alpha, and learning rates.
Larger Datasets: Use full datasets (e.g., FineTome-100k or HH-RLHF) with more computational resources.
Evaluation Metrics: Add quantitative evaluation steps to measure model performance post-fine-tuning.
Cross-Model Comparison: Compare DoRA, GRPO, and LoRA performance on the same dataset.

Conclusion
This repository provides a comprehensive pipeline for fine-tuning large language models using parameter-efficient methods. The code is optimized for memory-constrained environments and is suitable for researchers and practitioners looking to adapt LLMs for specific tasks.
License
This project is licensed under the MIT License. See the LICENSE file for details.
