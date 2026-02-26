import re


MAX_REWARD = 10

"""We create a regex format to match the reasoning sections and answers:"""
def get_match_format(reasoning_start = "<start_working_out>", reasoning_end = "<end_working_out>", solution_start = "<SOLUTION>", solution_end = "</SOLUTION>"):
    match_format = re.compile(
        rf"^[\s]{{0,}}"\
        rf"{reasoning_start}.+?{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    # """We verify it works:"""
    # match_format.search(
    #     "<start_working_out>Let me think!<end_working_out>"\
    #     "<SOLUTION>2</SOLUTION>",
    # )
    return match_format



"""We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:"""

def match_format_exactly(completions, **kwargs):
    scores = []
    match_format = get_match_format()
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""

def match_format_approximately(completions, reasoning_start = "<start_working_out>",
                                            reasoning_end   = "<end_working_out>",
                                            solution_start = "<SOLUTION>",
                                            solution_end = "</SOLUTION>", **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    match_format = get_match_format()

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores


def get_match_numbers(solution_start = "<SOLUTION>"):
    match_numbers = re.compile(
        solution_start + r".*?([\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )
    # print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
    # print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
    return match_numbers


"""Sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.
We also remove possible commas for example as in 123,456
"""

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    match_numbers = get_match_numbers()

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    
    # # Print only every few steps
    # global PRINTED_TIMES
    # global PRINT_EVERY_STEPS
    # if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
    #     print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    # PRINTED_TIMES += 1
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores