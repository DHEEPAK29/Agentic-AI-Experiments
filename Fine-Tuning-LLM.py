# the open-source checkpoint deepseek-ai/DeepSeek-R1-Distill-Qwen-7B and post-trains it with the Group Relative Policy Optimization (GRPO) objective you showed. 

# FLOW: 

# Loads the distilled Qwen-7B checkpoint.

# Samples G candidate answers for each prompt.

# Normalises reward scores inside each group to get advantages.

# Optimises the GRPO loss with clipping and KL regularisation.

# Saves a new policy ready for vLLM or SGLang inference.


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


# GRPO trainer implementation


import math, random, torch.nn.functional as F
from trl.trainer import PPOTrainer, PPOConfig

G               = 8          # group size
CLIP_EPS        = 0.2        # ε in the paper
KL_BETA         = 0.01       # β in the paper
MAX_TOKENS      = 512
BATCH_SIZE      = 16         # = groups per step

cfg = PPOConfig(batch_size=BATCH_SIZE, forward_batch_size=1,
                learning_rate=2e-5, log_with=None)

class GRPOTrainer(PPOTrainer):
    @torch.no_grad()
    def generate_group(self, prompts):
        """Generate G responses per prompt."""
        all_out, all_logprobs = [], []
        for _ in range(G):
            out = self.model.generate(
                **tok(prompts, return_tensors="pt", padding=True).to(self.model.device),
                max_new_tokens=MAX_TOKENS,
                do_sample=True, temperature=0.7, top_p=0.95, eos_token_id=tok.eos_token_id
            )
            texts = tok.batch_decode(out[:, :,], skip_special_tokens=True)
            # log probabilities for ratio πₜ/πₒₗd
            lp = self.model(**tok(texts, return_tensors="pt", padding=True).to(self.model.device),
                            labels=tok(texts, return_tensors="pt", padding=True)["input_ids"]).logits
            logps = -F.cross_entropy(lp[:, :-1].flatten(0,1),
                                     tok(texts, return_tensors="pt", padding=True)["input_ids"][:,1:].flatten(0,1),
                                     reduction="none").view(len(texts), -1).sum(1)
            all_out.append(texts);  all_logprobs.append(logps)
        return all_out, torch.stack(all_logprobs)

    def compute_loss(self, logps, old_logps, advantages, kl_term):
        ratio = torch.exp(logps - old_logps)
        unclipped = ratio * advantages
        clipped   = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages
        return -torch.mean(torch.min(unclipped, clipped) - KL_BETA * kl_term)

trainer = GRPOTrainer(cfg, policy, ref_policy, tok)



# Training loop

from torch.utils.data import DataLoader

def collate(batch): return {"prompt":[b["prompt"] for b in batch],
                            "responses":[b["responses"] for b in batch]}
loader = DataLoader(raw_ds.shuffle(seed=42), batch_size=BATCH_SIZE, collate_fn=collate)

for step, batch in enumerate(loader):
    # 1) Generate G responses
    outputs, logp_matrix = trainer.generate_group(batch["prompt"])  # shape = (G, B)
    # 2) Slice the reward scores in the same order
    score_matrix = torch.tensor([[r["score"] for r in resp] for resp in batch["responses"]]).T  # (G,B)

    # 3) Normalise per-group
    μ, σ = score_matrix.mean(0), score_matrix.std(0).clamp_min(1e-6)
    advantages = (score_matrix - μ) / σ

    # 4) Flatten to feed the optimiser
    loss = trainer.compute_loss(logp_matrix.flatten(),
                                logp_matrix.detach().flatten(),
                                advantages.flatten(),
                                (logp_matrix - trainer.ref_model_logps(outputs)).flatten())
    loss.backward();  trainer.optimizer.step();  trainer.optimizer.zero_grad()

    if step % 20 == 0:
        print(f"step {step:<6}  GRPO-loss = {loss.item():.4f}")


# Evaluation & saving

policy.save_pretrained("grpo-qwen7b-strategic")
tok.save_pretrained("grpo-qwen7b-strategic")

# quick sanity prompt
print(policy.generate(**tok("Explain pros & cons of hedging in supply-chain planning:",
                            return_tensors="pt").to(policy.device),
                      max_new_tokens=200, temperature=0.6))



