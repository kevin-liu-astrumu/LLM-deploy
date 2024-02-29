#!/bin/bash

# Set the desired port
PORT=5000
# Automatically detect the number of NVIDIA GPUs
TENSOR_PARALLEL_SIZE=$(nvidia-smi -L | wc -l)

# the bloke version is broken, find this relacement
# https://huggingface.co/casperhansen/mixtral-instruct-awq
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/mixtral-instruct-awq \
    --quantization awq \
    --dtype auto \
    --chat-template chat_templates/template_mistral.jinja \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE