import openai
def generate_cot_outline(question: str,
                          context: str = "",
                          model: str = "gpt-4",
                          n_steps: int = 4) -> str:
    """
    Ask an LLM to produce an n‑step Chain‑of‑Thought outline
    for the given question (and optional context).
    Returns a string like:
      Chain‑of‑Thought:
      1. …
      2. …
      …
    """
    prompt = (
        "You are a helpful reasoning assistant.\n"
        f"Generate a concise {n_steps}-step Chain-of-Thought outline\n"
        f"to answer the question below"
        + (f", given the context:\n\n{context}\n\n" if context else "\n\n")
        + f"Question: \"{question}\"\n\n"
        "Return only the numbered steps, e.g.:\n"
        "1. …\n"
        "2. …\n"
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system", "content":prompt}],
        temperature=0.3,
    )
    # assume the API returns exactly the steps we want
    steps = resp.choices[0].message.content.strip()
    return "Chain-of-Thought:\n" + steps
