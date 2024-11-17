from sac3 import paraphraser
from sac3.evaluator import Evaluate
from sac3.consistency_checker import SemanticConsistnecyCheck

# Input Information
# Define the test case: an original question and its target (expected) answer.
# These inputs are used to evaluate the SAC³ pipeline components.
question = 'Was there ever a US senator that represented the state of Alabama and whose alma mater was MIT?'
target_answer = 'Never'

# Step 1: Question Perturbation
# Generate semantically equivalent paraphrased questions.
# This tests the ability of the paraphraser module to create robust variations.
gen_question = paraphraser.paraphrase(
    question=question,
    number=3,              # Number of paraphrased questions to generate
    model='gpt-3.5-turbo', # LLM used for paraphrasing
    temperature=1.0        # High randomness for diverse paraphrases
)

# Step 2: LLM Evaluation
# Generate responses for the original and paraphrased questions using the LLM.
llm_evaluate = Evaluate(model='gpt-3.5-turbo')

# Generate responses for the original question
self_responses = llm_evaluate.self_evaluate(
    self_question=question, 
    temperature=1.0,       # High randomness for varied responses
    self_num=3             # Number of responses to generate
)

# Generate responses for the paraphrased (perturbed) questions
perb_responses = llm_evaluate.perb_evaluate(
    perb_questions=gen_question, 
    temperature=0.0        # Deterministic responses for consistency evaluation
)

# Step 3: Consistency Check
# Test the consistency of responses using the Semantic Consistency Checker (SCC).
scc = SemanticConsistnecyCheck(model='gpt-3.5-turbo')

# Check self-consistency: Evaluate consistency across responses to the same question.
sc2_score, sc2_vote = scc.score_scc(
    question=question, 
    target_answer=target_answer, 
    candidate_answers=self_responses, 
    temperature=0.0        # Deterministic evaluation
)
print("Self-consistency score:", sc2_score)
print("Self-consistency votes:", sc2_vote)

# Check perturbed question consistency: Evaluate consistency across responses to paraphrased questions.
sac3_q_score, sac3_q_vote = scc.score_scc(
    question=question, 
    target_answer=target_answer, 
    candidate_answers=perb_responses, 
    temperature=0.0        # Deterministic evaluation
)
print("Perturbed question consistency score:", sac3_q_score)
print("Perturbed question consistency votes:", sac3_q_vote)
