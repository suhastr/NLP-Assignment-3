from sac3 import llm_models

def paraphrase(question, number, model, temperature):
    """
    Generate semantically equivalent paraphrased versions of a given question.

    This function uses a language model (LLM) to generate multiple variations of
    the input question while preserving its semantic meaning. These variations
    are useful for robustness testing and consistency evaluation.

    Args:
        question (str): The original user query to be paraphrased.
        number (int): The number of paraphrased questions to generate.
        model (str): The LLM to use for generating paraphrases (e.g., GPT models).
        temperature (float): Sampling temperature for controlling randomness 
                             (typically 0 for deterministic paraphrasing).

    Returns:
        list: A list of `number` paraphrased questions that are semantically equivalent
              to the original question.
    """
    perb_questions = []  # List to store generated paraphrased questions

    # Construct the prompt for generating paraphrases
    prompt_temp = f'For question Q, provide {number} semantically equivalent questions.'
    prompt = prompt_temp + '\nQ:' + question

    # Call the LLM to generate the paraphrased questions
    res = llm_models.call_openai_model(prompt, model, temperature)
    
    # Split the response into individual questions
    res_split = res.split('\n')
    for line in res_split:
        perb_questions.append(line.strip())  # Add each paraphrased question to the list

    return perb_questions
