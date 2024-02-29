## VLLM model deployment 


To deploy a model, we can modify the deploy.config file. Config suggestions can be found [here](model_library/model.config)

Run the following command to deploy the model:

```bash
bash deploy.sh
```

You can find pre-defined bash files in `model_library` folder to deploy the LLM directly across all GPUs. The default internal port is `5000`.

```bash
source model_library/OpenOrca_mistral7B.sh
```

### Clear Cache Models

```bash
huggingface-cli delete-cache
```

## Embedding 

This folder holds notebooks of showing various embedding methods. 

