from tqdm import tqdm
import json

def adversarialize_question(question):
    """
    A simple adversarial transformation that appends a distractor clause.
    If question is a dict, it extracts the text from it (using key 'text').
    Modify the key as necessary to match your dataset structure.
    """
    # If the question is a dict, extract its text.
    if isinstance(question, dict):
        # Change "text" to the appropriate key if needed.
        question_text = question.get("text", str(question))
    else:
        question_text = question

    return question_text + " (consider an alternative perspective that challenges the obvious answer)"

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

    # Generate the adversarial version of the question
    adversarial_question = adversarialize_question(question)

    # Extract gold short answers only
    ground_truth_answers = []
    if "annotations" in example:
        ann = example["annotations"]
        if isinstance(ann, list):
            for annotation in ann:
                if isinstance(annotation, dict):
                    short_ans = annotation.get("short_answers", [])
                    # Only include examples if at least one short answer exists
                    if isinstance(short_ans, list) and len(short_ans) > 0:
                        for ans in short_ans:
                            if isinstance(ans, dict) and "text" in ans and ans["text"]:
                                print(ans)
                                ground_truth_answers.append(ans["text"][0])
                elif isinstance(annotation, str):
                    try:
                        annotation_dict = json.loads(annotation)
                        short_ans = annotation_dict.get("short_answers", [])
                        for ans in short_ans:
                            if isinstance(ans, dict) and "text" in ans and ans["text"]:
                                print(ans)
                                ground_truth_answers.append(ans["text"][0])
                    except Exception as e:
                        continue
        elif isinstance(ann, dict):
            short_ans = ann.get("short_answers", [])
            if isinstance(short_ans, list) and len(short_ans) > 0:
                for ans in short_ans:
                    if isinstance(ans, dict) and "text" in ans and ans["text"]:
                        print(ans)
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
average_tokens = total_tokens / processed_examples

print(f"Nova Pro Adversarial Evaluation on Natural Questions (first {processed_examples} examples):")
print(f"Exact Match: {overall_em:.2f}%")
print(f"F1 Score: {overall_f1:.2f}%")
print(f"Average token count per response: {average_tokens:.2f}")
