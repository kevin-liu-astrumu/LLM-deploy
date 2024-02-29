#!/bin/bash

# Set the desired port
PORT=5000
# Automatically detect the number of NVIDIA GPUs
TENSOR_PARALLEL_SIZE=$(nvidia-smi -L | wc -l)

python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/dolphin-2.7-mixtral-8x7b-AWQ \
    --quantization awq \
    --dtype auto \
    --chat-template chat_templates/template_chatml.jinja \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE