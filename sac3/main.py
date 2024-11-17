from sac3 import paraphraser
from sac3.evaluator import Evaluate
from sac3.consistency_checker import SemanticConsistnecyCheck

# Input Information
# Original question to evaluate and the target (expected) answer
question = 'Was there ever a US senator that represented the state of Alabama and whose alma mater was MIT?'
target_answer = 'Never'

# Step 1: Question Perturbation
# Generate semantically equivalent paraphrased versions of the original question.
# This helps evaluate the robustness of LLMs to variations in input phrasing.
gen_question = paraphraser.paraphrase(
    question, 
    number=3,              # Number of paraphrased versions to generate
    model='gpt-3.5-turbo', # LLM used for paraphrasing
    temperature=1.0        # High randomness for diverse paraphrasing
)

# Step 2: LLM Evaluation
# Evaluate the original and perturbed questions using an LLM to generate responses.
llm_evaluate = Evaluate(model='gpt-3.5-turbo')

# Generate responses for the original question
self_responses = llm_evaluate.self_evaluate(
    self_question=question, 
    temperature=1.0,       # High randomness for varied responses
    self_num=3             # Number of responses to generate
)

# Generate responses for the perturbed questions
perb_responses = llm_evaluate.perb_evaluate(
    perb_questions=gen_question, 
    temperature=0.0        # Deterministic responses for consistency evaluation
)

# Step 3: Consistency Check
# Use Semantic Consistency Checking (SCC) to evaluate consistency between responses.

# Initialize the Semantic Consistency Checker
scc = SemanticConsistnecyCheck(model='gpt-3.5-turbo')

# Consistency check for self-generated responses (same question)
# Measures the inconsistency (hallucination metric) across multiple responses to the same question.
sc2_score, sc2_vote = scc.score_scc(
    question=question, 
    target_answer=target_answer, 
    candidate_answers=self_responses, 
    temperature=0.0        # Deterministic evaluation
)
print("Self-consistency score:", sc2_score)
print("Self-consistency votes:", sc2_vote)

# Consistency check for perturbed questions
# Measures the inconsistency across responses to semantically equivalent questions.
sac3_q_score, sac3_q_vote = scc.score_scc(
    question=question, 
    target_answer=target_answer, 
    candidate_answers=perb_responses, 
    temperature=0.0        # Deterministic evaluation
)
print("Perturbed question consistency score:", sac3_q_score)
print("Perturbed question consistency votes:", sac3_q_vote)
