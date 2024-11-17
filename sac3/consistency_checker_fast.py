from sac3 import llm_models
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class SemanticConsistencyCheck:
    """
    A class to perform semantic consistency checks for hallucination detection 
    in language model (LLM) outputs, optimized for parallel execution.
    """

    def __init__(self, model):
        """
        Initialize the SemanticConsistencyCheck class.

        Args:
            model (str): The name or configuration of the language model to use.
        """
        self.model = model

        # Template prompt to guide LLM in evaluating semantic equivalence between two QA pairs.
        self.prompt_temp = """
        Are the following two Question-Answer(QA) pairs semantically equivalent? 
        Provide your best guess and the probability that it is correct (0.0 to 1.0).
        Given ONLY the guess (Yes or No) and probability, no other words or explanation. 
        For example:
        Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!> 
        Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; 
        just the probability!>
        """

    def openai_api_parallel(self, prompt, temperature):
        """
        Call the OpenAI API to evaluate semantic consistency for a single QA pair comparison.

        Args:
            prompt (str): The input prompt containing two QA pairs for comparison.
            temperature (float): Sampling temperature for controlling randomness in responses.

        Returns:
            str: The response from the OpenAI API.
        """
        return llm_models.call_openai_model(prompt, self.model, temperature)

    def score_scc_api(self, question, target_answer, candidate_answers, temperature):
        """
        Compute the semantic consistency score between a target QA pair and multiple candidate QA pairs
        using parallelized API calls.

        Args:
            question (str): The original user query.
            target_answer (str): The primary response to the query (baseline).
            candidate_answers (list): List of candidate responses to compare against the target.
            temperature (float): Sampling temperature for response generation.

        Returns:
            tuple:
                - score (float): Average inconsistency score (hallucination metric).
                - sc_output (list): Binary consistency results for each candidate (0 = consistent, 1 = inconsistent).
        """
        if target_answer is None:
            raise ValueError("Target answer cannot be None.")

        sc_output = []  # List to store binary consistency results
        target_pair = f'Q:{question}\nA:{target_answer}'  # Format the target QA pair
        num_candidate_answer = len(candidate_answers)

        # Use ThreadPoolExecutor to execute API calls in parallel
        with ThreadPoolExecutor(max_workers=num_candidate_answer) as executor:
            all_res = []

            # Submit parallel tasks to compare target QA with each candidate QA
            for candidate in candidate_answers:
                candidate_pair = f'Q:{question}\nA:{candidate}'
                prompt = (self.prompt_temp + 
                          '\nThe first QA pair is:\n' + target_pair + 
                          '\nThe second QA pair is:\n' + candidate_pair)
                all_res.append(executor.submit(self.openai_api_parallel, prompt, temperature))

            # Process completed tasks and collect results
            for temp in concurrent.futures.as_completed(all_res):
                res = temp.result()
                guess = res.split(':')[1].split('\n')[0].strip()  # Extract "Guess" (Yes/No)
                value = 0 if guess == 'Yes' else 1  # Map "Yes" to 0 and "No" to 1
                sc_output.append(value)

        # Compute the average inconsistency score
        score = sum(sc_output) / num_candidate_answer
        return score, sc_output

    def score_scc(self, question, target_answer, candidate_answers, temperature):
        """
        Compute the semantic consistency score sequentially (non-parallel version).

        Args:
            question (str): The original user query.
            target_answer (str): The primary response to the query (baseline).
            candidate_answers (list): List of candidate responses to compare against the target.
            temperature (float): Sampling temperature for response generation.

        Returns:
            tuple:
                - score (float): Average inconsistency score (hallucination metric).
                - sc_output (list): Binary consistency results for each candidate (0 = consistent, 1 = inconsistent).
        """
        if target_answer is None:
            raise ValueError("Target answer cannot be None.")

        sc_output = []  # List to store binary consistency results
        target_pair = f'Q:{question}\nA:{target_answer}'  # Format the target QA pair
        num_candidate_answer = len(candidate_answers)

        # Sequentially compare target QA with each candidate QA
        for candidate in candidate_answers:
            candidate_pair = f'Q:{question}\nA:{candidate}'
            prompt = (self.prompt_temp + 
                      '\nThe first QA pair is:\n' + target_pair + 
                      '\nThe second QA pair is:\n' + candidate_pair)

            # Call the OpenAI API for semantic equivalence evaluation
            res = llm_models.call_openai_model(prompt, self.model, temperature)
            guess = res.split(':')[1].split('\n')[0].strip()  # Extract "Guess" (Yes/No)
            value = 0 if guess == 'Yes' else 1  # Map "Yes" to 0 and "No" to 1
            sc_output.append(value)

        # Compute the average inconsistency score
        score = sum(sc_output) / num_candidate_answer
        return score, sc_output
