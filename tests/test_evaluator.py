from sac3 import paraphraser
from sac3.evaluator import Evaluate

# Input Information
# Define the test case: an original question to evaluate.
# This test checks the LLM's response to the question and its perturbed versions.
question = 'Is pi smaller than 3.2?'

# Step 1: Question Perturbation
# Generate semantically equivalent paraphrased versions of the question.
# This tests the paraphraser module's ability to generate meaningful variations.
gen_question = paraphraser.paraphrase(
    question=question, 
    number=5,              # Number of paraphrased questions to generate
    model='gpt-3.5-turbo', # LLM used for paraphrasing
    temperature=1.0        # High randomness for diverse paraphrases
)

# Step 2: LLM Evaluation
# Evaluate the original question and its perturbed versions using an LLM.
llm_evaluate = Evaluate(model='gpt-3.5-turbo')

# Generate responses for the original question
self_responses = llm_evaluate.self_evaluate(
    self_question=question, 
    temperature=1.0,       # High randomness for varied responses
    self_num=5             # Number of responses to generate for the original question
)

# Generate responses for the perturbed (paraphrased) questions
perb_responses = llm_evaluate.perb_evaluate(
    perb_questions=gen_question, 
    temperature=0.0        # Deterministic responses for consistency evaluation
)

# Step 3: Output Results
# Print the original question, the generated responses for it, and the responses for the perturbed questions.
print('Original question:', question)
print('Generated responses (self_responses):', self_responses)
print('Generated responses for perturbed questions (perb_responses):', perb_responses)
