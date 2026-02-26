
from datasets import load_dataset

def prep_dataset(dataset_name, reasoning_start = "<start_working_out>",
                                reasoning_end   = "<end_working_out>",
                                solution_start = "<SOLUTION>",
                                solution_end = "</SOLUTION>"):
    dataset = None
    if dataset_name=="GSM8K" or dataset_name=="gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split = "train")
    system_prompt = get_system_prompt(reasoning_start, reasoning_end, solution_start, solution_end)
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    return dataset


def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()


def get_system_prompt(reasoning_start, reasoning_end, solution_start, solution_end):
    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""
    return system_prompt


