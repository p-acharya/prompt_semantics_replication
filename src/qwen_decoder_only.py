import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, Qwen2Tokenizer

import wandb

directory = os.path.dirname(os.getcwd())
test_mode = False
if test_mode:
    entity = "pacharya-george-mason-university"
    project = "qwen-rte-experiment"
    # weave.init(f"{entity}/{project}")
else:
    entity = "gmu-nlp"
    project = "prompt-semantics-replication"


model_names = {
    "qwen-7b-base": "/projects/antonis/models/Qwen2.5/Qwen2.5-7B",
    "qwen-7b-instruct": "/projects/antonis/models/Qwen2.5-7B-Instruct",
}


# @weave.op()
def classify_response(response_text):
    """
    Classifies a model's response into one of three categories:
    - 0: Entailment
    - 1: Not Entailment
    - -1: Invalid / Malformed
    """
    response_lower = response_text.lower()

    # Check for a clear "entailment" prediction
    is_entailment = (
        "entailment" in response_lower and "not" not in response_lower
    )

    # Check for a clear "not entailment" prediction
    is_not_entailment = "not entailment" in response_lower

    # Use elif to ensure mutual exclusivity
    if is_entailment:
        return 0
    elif is_not_entailment:
        return 1
    else:
        # If neither of the above, the response is invalid for our task
        return -1

def classify_response_v3(response_text):
    """
    V3: A highly robust classifier that handles:
    - Standard keywords ("entailment", "not entailment")
    - Natural language variations ("is correct", "is false")
    - Simple binary answers ("Yes", "No")
    - Prompt echoing ("Answer: entailment")
    - 0: Entailment
    - 1: Not Entailment
    - -1: Invalid / Malformed
    """
    response_lower = response_text.lower().strip()
    
    # --- Keywords for Not Entailment ---
    # We check for these first as they are less ambiguous.
    not_entailment_keywords = [
        "not entailment",
        "does not entail",
        "is not correct",
        "is false",
        "is incorrect",
        "cannot be inferred",
        "no"  # Added "no"
    ]
    
    # --- Keywords for Entailment ---
    entailment_keywords = [
        "entailment",
        "is correct",
        "is true",
        "can be inferred",
        "yes" # Added "yes"
    ]
    
    # --- Logic ---
    # Check for non-entailment first, as it's a stronger signal.
    # This correctly handles cases like "Yes, but this is not entailment."
    if any(keyword in response_lower for keyword in not_entailment_keywords):
        return 1
        
    # Then check for entailment, ensuring no negations are present.
    if any(keyword in response_lower for keyword in entailment_keywords) and "not" not in response_lower:
        return 0
        
    # If neither of the above, the response is still considered invalid
    return -1

def run_experiment():
    # --- 0. Check for GPU ---
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This script requires a GPU, but CUDA is not available."
        )

    # --- 1. Configuration ---
    model_name = model_names["qwen-7b-base"]
    prompt_path = "/scratch/pachary4/functional_competence/prompt_semantics/data/binary_NLI_prompts.csv"
    dataset_name = "rte"
    num_shots_list, seeds = [32], [1]
    # num_shots_list = [0, 4, 16]
    # seeds = [1, 2, 3]  # Use multiple seeds for robustness
    max_new_tokens = 10

    job_id = os.environ.get("SLURM_JOB_ID")
    gpu_type = torch.cuda.get_device_name(0)

    # --- 2. Load Model and Tokenizer ---
    # The from_pretrained method can handle both local paths and Hugging Face Hub IDs.
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded on device: {model.device}")

    # --- 3. Load Dataset and Prompts ---
    dataset = load_dataset("glue", dataset_name)
    prompts_df = pd.read_csv(prompt_path)
    # The original experiment has different prompt groups; we'll use 'sec4' as an example
    prompts_to_test = prompts_df[prompts_df["experiment"] == "sec4"]

    results = []

    if test_mode:
        print("--- RUNNING IN QUICK TEST MODE ---")
        prompts_to_test = prompts_to_test.tail(3).head(2)
        dataset["validation"] = dataset["validation"].select(range(5))
        num_shots_list = [4, 16]
        seeds = [1]

    wandb.init(
        entity=entity,
        project=project,
        config={
            "model_name": os.path.basename(model_name),
            "dataset_name": dataset_name,
            "prompt_path": os.path.basename(prompt_path),
            "num_shots_list": num_shots_list,
            "seeds": seeds,
            "slurm_job_id": job_id,
            "gpu": gpu_type,
            "max_new_tokens": max_new_tokens,
        },
    )

    individual_results = []

    # --- 4. Run Experiments ---
    for _, prompt_row in tqdm(
        list(prompts_to_test.iterrows()), desc="Prompts"
    ):
        prompt_template = prompt_row["template"]
        for num_shots in tqdm(num_shots_list, desc="Shots"):
            for seed in tqdm(seeds, desc="Seeds"):
                # --- 5. Create Few-Shot Examples ---
                random.seed(seed)
                # Ensure the number of shots does not exceed the available training examples
                if num_shots > len(dataset["train"]):
                    print(
                        f"Warning: num_shots ({num_shots}) is greater than the number of training examples ({len(dataset['train'])}). Skipping."
                    )
                    continue

                train_examples = random.sample(
                    list(dataset["train"]), num_shots
                )

                correct_predictions = 0
                invalid_responses = 0
                total_predictions = 0

                for eval_example in tqdm(
                    dataset["validation"], desc="Evaluating"
                ):
                    # --- 6. Construct the Prompt ---
                    context = ""
                    for train_ex in train_examples:
                        # The answer is 'entailment' (0) or 'not_entailment' (1)
                        answer = (
                            "entailment"
                            if train_ex["label"] == 0
                            else "not entailment"
                        )
                        # Use the prompt_template to format each few-shot example
                        example_text = prompt_template.format(
                            premise=train_ex["sentence1"],
                            hypothesis=train_ex["sentence2"],
                        )
                        context += f"{example_text} Answer: {answer}\n\n"

                    # Use the prompt_template to format the final evaluation example
                    eval_text = prompt_template.format(
                        premise=eval_example["sentence1"],
                        hypothesis=eval_example["sentence2"],
                    )
                    prompt = f"{context}{eval_text}"

                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ]

                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = tokenizer(
                        [text], return_tensors="pt", padding=True
                    ).to(model.device)

                    # --- 7. Generate and Evaluate ---
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                    )
                    generated_ids = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(
                            model_inputs.input_ids, generated_ids
                        )
                    ]

                    response = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                    # Use the new, more robust classification function
                    predicted_label = classify_response(response)
                    true_label = eval_example["label"]

                    adherent = predicted_label != -1
                    correct = adherent and (predicted_label == true_label)

                    if adherent:
                        if correct:
                            correct_predictions += 1
                    else:
                        invalid_responses += 1

                    total_predictions += 1

                    individual_results.append(
                        {
                            "prompt": prompt,
                            "num_shots": num_shots,
                            "prompt_template": prompt_template,
                            "premise": eval_example["sentence1"],
                            "hypothesis": eval_example["sentence2"],
                            "true_label": true_label,
                            "response": response,
                            "predicted_label": predicted_label,
                            "adherent": adherent,
                            "correct": correct if adherent else None,
                        }
                    )

                # --- After the loop, you can calculate much better metrics ---
                adherence_rate = (
                    (total_predictions - invalid_responses) / total_predictions
                    if total_predictions > 0
                    else 0
                )
                if (total_predictions - invalid_responses) > 0:
                    accuracy_on_valid = correct_predictions / (
                        total_predictions - invalid_responses
                    )
                else:
                    accuracy_on_valid = 0.0  # Avoid division by zero if all responses are invalid

                print(
                    f"Prompt: '{prompt_template}', Shots: {num_shots}, Seed: {seed}, "
                    f"Adherence: {adherence_rate:.4f}, Accuracy on Valid: {accuracy_on_valid:.4f}"
                )
                results.append(
                    {
                        "prompt": prompt_template,
                        "num_shots": num_shots,
                        "seed": seed,
                        "adherence_rate": adherence_rate,
                        "accuracy_on_valid": accuracy_on_valid,
                        "invalid_responses": invalid_responses,
                    }
                )
                wandb.log(
                    {
                        "prompt": prompt_template,
                        "num_shots": num_shots,
                        "seed": seed,
                        "adherence_rate": adherence_rate,
                        "accuracy_on_valid": accuracy_on_valid,
                        "invalid_responses": invalid_responses,
                        "individual_results": wandb.Table(
                            dataframe=pd.DataFrame(individual_results)
                        ),
                    }
                )

    # --- 8. Save and Analyze Results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{job_id}-qwen_rte_results.csv", index=False)
    wandb.log({"summary_results": wandb.Table(dataframe=results_df)})
    print(f"\nResults saved to {job_id}-qwen_rte_results.csv")
    # print("\n--- Average Accuracy per Prompt ---")
    # print(results_df.groupby("prompt")["accuracy"].mean())
    wandb.finish()


if __name__ == "__main__":
    run_experiment()
