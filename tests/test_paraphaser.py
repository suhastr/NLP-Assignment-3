from sac3 import paraphraser

# Input Information
# Define the test case: an original question to paraphrase.
# This test checks the ability of the paraphraser to generate semantically equivalent variations.
question = 'Which city is the capital of Maryland?'

# Step 1: Generate Question Perturbations
# Use the paraphraser module to create multiple semantically equivalent paraphrased questions.
gen_question = paraphraser.paraphrase(
    question=question, 
    number=5,              # Number of paraphrased questions to generate
    model='gpt-3.5-turbo', # LLM used for generating paraphrases
    temperature=1.0        # High randomness for diverse paraphrasing
)

# Step 2: Output Results
# Print the original question and the generated paraphrased questions for review.
print('Original question:', question)
print('Generated paraphrased questions:', gen_question)
