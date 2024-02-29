#!/bin/bash


# Set the desired port
PORT=5000
# Automatically detect the number of NVIDIA GPUs
TENSOR_PARALLEL_SIZE=$(nvidia-smi -L | wc -l)

# Use the detected number of GPUs for the tensor parallel size in the command

python -m vllm.entrypoints.openai.api_server \
    --model teknium/OpenHermes-2.5-Mistral-7B \
    --chat-template chat_templates/template_chatml.jinja \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE