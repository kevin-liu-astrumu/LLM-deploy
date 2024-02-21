source deploy.config
python -m vllm.entrypoints.openai.api_server \
        --model $model \
        --quantization $quantization \
        --dtype $dtype \
        --chat-template $chat_template \
        --port $port \
        --tensor-parallel-size $tensor_parallel_size