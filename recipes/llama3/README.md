# Llama3

---
**NOTE**
To make your life easier, run these commands from the recipe directory (here `recipes/llama3`).
---

## Retrieve and convert model

### Set environment variables

```bash
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```bash
eole convert HF --model_dir meta-llama/Meta-Llama-3-8B-Instruct --output $EOLE_MODEL_DIR/llama3-8b-instruct --token $HF_TOKEN
```

## Inference

### Write test prompt to text file

#### Simple prompt

```bash
echo -e "What are some nice places to visit in France?" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
```

#### Structured prompt

Full prompt with special tokens should also be handled properly:

```bash
echo -e "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>" | sed ':a;N;$!ba;s/\n/｟newline｠/g' > test_prompt.txt
```

### Run inference

```bash
eole predict -c llama-inference.yaml -src test_prompt.txt -output test_output.txt
```