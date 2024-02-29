#!/bin/bash


# Set the desired port
PORT=5000
# Automatically detect the number of NVIDIA GPUs
TENSOR_PARALLEL_SIZE=$(nvidia-smi -L | wc -l)

# good model, no quantization
python -m vllm.entrypoints.openai.api_server \
    --model Open-Orca/Mistral-7B-OpenOrca \
    --chat-template chat_templates/template_chatml.jinja \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE