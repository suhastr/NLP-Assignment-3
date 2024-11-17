import llm_models
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class SemanticConsistencyCheck:
    """
    A class to perform semantic consistency checks for hallucination detection in black-box LMs.

    This class evaluates the consistency between a target QA pair and candidate QA pairs using 
    semantically-aware prompts, leveraging parallelized calls to LLM APIs for efficiency.
    """

    def __init__(self, model):
        """
        Initialize the SemanticConsistencyCheck class.

        Args:
            model (str): The name or configuration of the language model to be used for evaluation.
        """
        self.model = model
        
        # Template prompt to guide LLM in comparing QA pairs for semantic equivalence.
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
        Call the OpenAI API for semantic consistency check in a parallelized manner.

        Args:
            prompt (str): The input prompt to be passed to the LLM.
            temperature (float): Sampling temperature to control randomness in LLM responses.

        Returns:
            str: The response from the LLM API.
        """
        # Call the LLM API with the given prompt and model configuration.
        return llm_models.call_openai_model(prompt, self.model, temperature)

    def score_scc_api(self, question, target_answer, candidate_answers, temperature):
        """
        Compute the semantic consistency score between a target QA pair and multiple candidate QA pairs 
        using multithreaded API calls for parallel execution.

        Args:
            question (str): The original user query.
            target_answer (str): The primary response to the query (baseline).
            candidate_answers (list): List of candidate responses generated for perturbed versions of the query.
            temperature (float): Sampling temperature for response generation.

        Returns:
            tuple: 
                - score (float): Average inconsistency score across all candidate answers.
                - sc_output (list): Binary consistency results for each candidate answer (0 = consistent, 1 = inconsistent).
        """
        if target_answer is None:
            raise ValueError("Target answer cannot be None. ")

        # Prepare the target QA pair for comparison.
        sc_output = []
        target_pair = 'Q:' + question + '\nA:' + target_answer
        num_candidate_answer = len(candidate_answers)

        # Use a ThreadPoolExecutor to parallelize LLM API calls.
        with ThreadPoolExecutor(max_workers=num_candidate_answer + 2) as executor:
            all_res = []
            for i in range(num_candidate_answer):
                # Construct the prompt for comparing the target QA with each candidate QA.
                candidate_pair = 'Q:' + question + '\nA:' + candidate_answers[i]
                prompt = self.prompt_temp + '\nThe first QA pair is:\n' + target_pair + '\nThe second QA pair is:\n' + candidate_pair
                
                # Submit the API call as a separate thread for execution.
                output = executor.submit(self.openai_api_parallel, prompt, temperature)
                all_res.append(output)

            # Collect and process the results from completed threads.
            for temp in concurrent.futures.as_completed(all_res):
                res = temp.result()
                guess = res.split(':')[1].split('\n')[0].strip()  # Extract the guess (Yes/No).
                value = 0 if guess == 'Yes' else 1  # Map Yes/No to binary values.
                sc_output.append(value)

        # Calculate the inconsistency score as the average of binary values.
        score = sum(sc_output) / num_candidate_answer
        return score, sc_output

    def score_scc(self, question, target_answer, candidate_answers, temperature):
        """
        Sequential implementation of semantic consistency check (non-parallel version).

        Args:
            question (str): The original user query.
            target_answer (str): The primary response to the query (baseline).
            candidate_answers (list): List of candidate responses generated for perturbed versions of the query.
            temperature (float): Sampling temperature for response generation.

        Returns:
            tuple: 
                - score (float): Average inconsistency score across all candidate answers.
                - sc_output (list): Binary consistency results for each candidate answer (0 = consistent, 1 = inconsistent).
        """
        if target_answer is None:
            raise ValueError("Target answer cannot be None. ")

        # Prepare the target QA pair for comparison.
        sc_output = []
        target_pair = 'Q:' + question + '\nA:' + target_answer
        num_candidate_answer = len(candidate_answers)

        # Iterate through candidate answers and compare each with the target QA pair.
        for i in range(num_candidate_answer):
            candidate_pair = 'Q:' + question + '\nA:' + candidate_answers[i]
            prompt = self.prompt_temp + '\nThe first QA pair is:\n' + target_pair + '\nThe second QA pair is:\n' + candidate_pair
            
            # Call the LLM API to evaluate semantic equivalence.
            res = llm_models.call_openai_model(prompt, self.model, temperature)
            guess = res.split(':')[1].split('\n')[0].strip()  # Extract the guess (Yes/No).
            value = 0 if guess == 'Yes' else 1  # Map Yes/No to binary values.
            sc_output.append(value)

        # Calculate the inconsistency score as the average of binary values.
        score = sum(sc_output) / num_candidate_answer
        return score, sc_output
