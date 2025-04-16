from tqdm import tqdm
import json

def adversarialize_question_with_chain_of_thought(question):
    """
    Generates an adversarial version of the question along with a chain-of-thought (CoT)
    that explains the reasoning process for the adversarial transformation.
    
    If question is a dict, it extracts its text using key 'text'. Modify the key as needed.
    """
    # Extract text from question (if it's a dict) or use it directly.
    if isinstance(question, dict):
        question_text = question.get("text", str(question))
    else:
        question_text = question

    # Chain of thought steps (this is a simplified example)
    chain_of_thought = (
        "Step 1: Identify the main focus and assumptions in the original question. "
        "Step 2: Recognize that the question might be interpreted narrowly. "
        "Step 3: Introduce a perspective that challenges the obvious interpretation by "
        "prompting the consideration of alternative viewpoints."
    )

    # Create adversarial question by appending a distractor clause
    adversarial_question = question_text + " (consider an alternative perspective that challenges the obvious answer)"

    return chain_of_thought, adversarial_question

total_em = 0
total_f1 = 0
total_tokens = 0
processed_examples = 0
max_examples = 5

for example in tqdm(nq_dataset, desc="Adversarial Evaluation on Natural Questions"):
    if processed_examples >= max_examples:
        break

    question = example["question"]
    context = example.get("document_text", "")

    # Generate the adversarial version of the question with chain-of-thought
    chain_of_thought, adversarial_question = adversarialize_question_with_chain_of_thought(question)
    
    print("Chain-of-Thought Explanation:")
    print(chain_of_thought)
    print("-----")
    
    # Extract gold short answers only
    ground_truth_answers = []
    if "annotations" in example:
        ann = example["annotations"]
        if isinstance(ann, list):
            for annotation in ann:
                if isinstance(annotation, dict):
                    short_ans = annotation.get("short_answers", [])
                    if isinstance(short_ans, list) and len(short_ans) > 0:
                        for ans in short_ans:
                            if isinstance(ans, dict) and "text" in ans and ans["text"]:
                                print("Gold Answer Snippet:", ans)
                                ground_truth_answers.append(ans["text"][0])
                elif isinstance(annotation, str):
                    try:
                        annotation_dict = json.loads(annotation)
                        short_ans = annotation_dict.get("short_answers", [])
                        for ans in short_ans:
                            if isinstance(ans, dict) and "text" in ans and ans["text"]:
                                print("Gold Answer Snippet:", ans)
                                ground_truth_answers.append(ans["text"][0])
                    except Exception as e:
                        continue
        elif isinstance(ann, dict):
            short_ans = ann.get("short_answers", [])
            if isinstance(short_ans, list) and len(short_ans) > 0:
                for ans in short_ans:
                    if isinstance(ans, dict) and "text" in ans and ans["text"]:
                        print("Gold Answer Snippet:", ans)
                        ground_truth_answers.append(ans["text"][0])
    
    print("Ground Truth Answers:", ground_truth_answers)
    if not any(ans.strip() for ans in ground_truth_answers):
        continue

    # Use the adversarial question for querying the model
    pred_answer = query_nova_pro(adversarial_question, context)

    print("Original Question:", question)
    print("Adversarial Question:", adversarial_question)
    print("Dataset Answer:", ground_truth_answers)
    print("Nova Pro Output:", pred_answer)

    # Compute Exact Match and F1 scores against all ground truth answers
    em_scores = [compute_exact(gt_ans, pred_answer) for gt_ans in ground_truth_answers]
    f1_scores = [compute_f1(gt_ans, pred_answer) for gt_ans in ground_truth_answers]

    best_em = max(em_scores)
    best_f1 = max(f1_scores)

    total_em += best_em
    total_f1 += best_f1
    processed_examples += 1

    print("\n\n")

overall_em = (total_em / processed_examples) * 100
overall_f1 = (total_f1 / processed_examples) * 100
average_tokens = total_tokens / processed_examples if processed_examples > 0 else 0

print(f"Nova Pro Adversarial Evaluation on Natural Questions (first {processed_examples} examples):")
print(f"Exact Match: {overall_em:.2f}%")
print(f"F1 Score: {overall_f1:.2f}%")
print(f"Average token count per response: {average_tokens:.2f}")
