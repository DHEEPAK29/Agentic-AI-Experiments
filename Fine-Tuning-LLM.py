# the open-source checkpoint deepseek-ai/DeepSeek-R1-Distill-Qwen-7B and post-trains it with the Group Relative Policy Optimization (GRPO) objective you showed. 

# conda create -n grpo-qwen-7b python=3.10 -y
# conda activate grpo-qwen-7b

# Core DL stack
# pip install "torch>=2.2" "transformers>=4.40" accelerate bitsandbytes
# # RL + helpers
# pip install trl==0.8.6 datasets peft evaluate tqdm
# # (optional) flash-attention2 for memory-efficient training
# pip install flash-attn --no-build-isolation

# Hardware note – with 4-bit quantisation, gradient checkpointing and Flash-Attention, a single 80 GB A100 can train the 7 B model at a global batch = 128 (8 × 16). On smaller GPUs use torchrun --fsdp … or DeepSpeed ZeRO-3.  

# {"prompt": "...", "responses": [{"text": "...", "score": 7.8}, ...]}  

from datasets import load_dataset, Dataset
raw_ds = load_dataset("json", data_files="strategic_decision_data.jsonl")["train"]

# Sanity-check format
print(raw_ds[0])

# Load policy, reference and tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, copy

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
policy = AutoModelForCausalLM.from_pretrained(model_name,
                                              quantization_config=bnb,
                                              torch_dtype=torch.bfloat16,
                                              device_map="auto")

# A frozen reference network for the KL term
ref_policy = copy.deepcopy(policy).eval()
for p in ref_policy.parameters(): p.requires_grad_(False)
