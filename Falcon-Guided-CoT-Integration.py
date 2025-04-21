from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

# 1. Load a free, open‑access instruction‑tuned model
model_name = "tiiuae/falcon-7b-instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. Custom stopping to cut off after n numbered steps
class StopOnNumberedSteps(StoppingCriteria):
    def __init__(self, tokenizer, max_steps=3):
        self.tokenizer = tokenizer
        self.max_steps = max_steps
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # stop once we've seen "n." n times (plus the initial newline)
        return text.count("\n") > self.max_steps

def generate_cot_outline(question: str,
                         context: str = "",
                         n_steps: int = 4,
                         max_new_tokens: int = 200,
                         temperature: float = 0.3) -> str:
    prompt = (
        "You are a helpful reasoning assistant.\n"
        f"Generate a concise {n_steps}-step Chain-of-Thought outline to answer the question below"
        + (f", given the context:\n{context}\n" if context else "\n")
        + f"Question: \"{question}\"\n\n"
        "Return only the numbered steps, e.g.:\n"
        "1. …\n"
        "2. …\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopping = StoppingCriteriaList([StopOnNumberedSteps(tokenizer, max_steps=n_steps)])
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stopping_criteria=stopping,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip off anything before the first "1."
    if "1." in gen:
        gen = gen.split("1.", 1)[1]
    lines = gen.strip().split("\n")[:n_steps]
    numbered = "\n".join(f"{i+1}. {line.strip().lstrip('0123456789. ')}"
                         for i, line in enumerate(lines))
    return "Chain-of-Thought:\n" + numbered

