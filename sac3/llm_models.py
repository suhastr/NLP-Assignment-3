import os
# Uncomment and update your cache directory for Hugging Face if needed
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/ec2-user/SageMaker/hf_cache'
# os.environ['HF_HOME'] = '/home/ec2-user/SageMaker/hf_cache'

import openai
import torch
from peft import PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Initialize OpenAI API
# Replace with your OpenAI API key
# openai.api_key = 'your openai key'

def call_openai_model(prompt, model, temperature):
    """
    Calls the OpenAI ChatCompletion API to generate a response.

    Args:
        prompt (str): The input prompt/question for the model.
        model (str): The OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4').
        temperature (float): Sampling temperature to control randomness in responses.

    Returns:
        str: The generated response from the OpenAI model.
    """
    response = None
    while response is None:  # Retry mechanism for robust API calls
        try:
            # OpenAI API call to generate a response
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )
        except Exception as e:
            # Handle specific errors and implement retry logic
            if 'is greater than the maximum' in str(e):
                raise BatchSizeException()  # Custom exception for batch size errors
            print(e)
            print('Retrying...')
            time.sleep(2)
        try:
            # Extract and return the generated response
            output = response.choices[0].message.content
        except Exception:
            # Handle cases where no valid response is generated
            output = 'do not have response from chatgpt'
    return output


def call_guanaco_33b(prompt, max_new_tokens):
    """
    Calls the Guanaco-33B model to generate a response.

    Args:
        prompt (str): The input prompt/question for the model.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated response from the Guanaco-33B model.
    """
    # Define model and adapter names
    model_name = "huggyllama/llama-30b"
    adapters_name = 'timdettmers/guanaco-33b'
    
    # Load the base model and adapter with specific configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={i: '16384MB' for i in range(torch.cuda.device_count())},  # Optimized for GPUs with 16GB memory
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Format the prompt for the conversation
    formatted_prompt = (
        f"A chat between a curious human and an artificial intelligence assistant."
        f"The assistant gives helpful, concise, and polite answers to the user's questions.\n"
        f"### Human: {prompt} ### Assistant:"
    )
    
    # Tokenize the input and move it to the appropriate device
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
    
    # Generate the model's output
    outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the relevant portion of the response
    res_sp = res.split('###')
    output = res_sp[1] + res_sp[2]
    
    return output


def call_falcon_7b(prompt, max_new_tokens):
    """
    Calls the Falcon-7B model to generate a response.

    Args:
        prompt (str): The input prompt/question for the model.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated response from the Falcon-7B model.
    """
    # Define the model and tokenizer
    model = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Initialize the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Generate text using the pipeline
    sequences = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # Enable sampling for varied responses
        top_k=10,        # Restrict output to top 10 tokens for diversity
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Extract the generated text from the response
    for seq in sequences:
        res = seq['generated_text']

    return res
