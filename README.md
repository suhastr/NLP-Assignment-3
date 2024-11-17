Below are the orginial Paper and Orginial work I have cloned it to reproduce the same for my class assignment. 
Paper : SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency
Link: https://aclanthology.org/2023.findings-emnlp.1032/
Orginial GitHub Code Link: https://github.com/intuit/sac3/tree/main


# **Notebook Title: SAC3 Methodology Implementation and Robustness Testing using CheckList**
---

## **Overview**
This notebook evaluates the robustness and reliability of language models using a combination of **CheckList** and **SAC3** frameworks. It includes behavioral testing for linguistic phenomena and hallucination detection in factual responses. The tests implemented here align with state-of-the-art benchmarks and are inspired by the following research papers:
- **CheckList**: Beyond Accuracy: Behavioral Testing of NLP Models (ACL 2020).
- **SAC3**: Reliable Hallucination Detection in Black-Box Language Models (EMNLP 2023).

---

## **Purpose**
- To systematically evaluate model performance under various linguistic perturbations.
- To detect hallucinations and inconsistencies in fact-based tasks.
- To explore the alignment between tests from CheckList and SAC3.

---

## **Table of Contents**
1. [Introduction](#Introduction)
2. [Datasets](#Datasets)
   - [Dataset Descriptions](#Dataset-Descriptions)
   - [Dataset Preprocessing](#Dataset-Preprocessing)
3. [CheckList Test](#Testing-Methodology)
   - [Negation](#Negation)
   - [Coreference Resolution](#Coreference-Resolution)
   - [Temporal Reasoning](#Temporal-Reasoning)
   - [Semantic Role Labeling](#Semantic-Role-Labeling)
   - [Fairness and Bias](#Fairness-and-Bias)
   - [Robustness (Typographical Errors)](#Robustness-Typographical-Errors)
   - [Vocabulary and Synonym Substitution](#Vocabulary-and-Synonym-Substitution)
4. [Results](#Results)
   - [Behavioral Test Results](#Behavioral-Test-Results)
   - [Hallucination Detection Results](#Hallucination-Detection-Results)
   - [Generated Tables](#Generated-Tables)
5. [Conclusion](#Conclusion)
6. [References](#References)

---

## **Section Details**

### **1. Introduction**
- In this notebook, I have attempted to replicate the implementation of the SAC3 methodology as described in the paper and have also incorporated selected robustness tests from the CheckList framework.


---

### **2. Datasets**
#### **Dataset Descriptions**
### SAC3 Experiments and Datasets

### Note on Dataset Usage
- In this notebook, **minimal datasets** were used instead of the original datasets (Prime and Senator) due to **computational cost and resource constraints**.
- Results may differ from the paper's reported findings as the reduced dataset size impacts accuracy and consistency metrics.

### Datasets Used in SAC3 (from the Paper)

### 1. **Prime Number Dataset**
- **Description**: Contains 500 questions querying the primality of randomly chosen numbers between 1,000 and 20,000.
- **Factual Answers**: Always "Yes" (all numbers are prime).
- **Hallucinated Answers**: "No, it is not a prime number."
- **Purpose**: Evaluates consistency in identifying factual and hallucinated responses in binary classification tasks.

### 2. **Senator Search Dataset**
- **Description**: Comprises 500 questions structured as:
  - _"Was there ever a US senator that represented [STATE] and whose alma mater was [COLLEGE]?"_
- **Factual Answers**: Always "No" (no such senator exists).
- **Hallucinated Answers**: "Yes, there was a senator who met the criteria."
- **Purpose**: Tests model reliability in handling hypothetical and factual queries.

### Relevance:
- These datasets are extensively used in **Table 3, Table 4, and Table 5** of the paper for evaluating hallucination detection methods.




---

### **3. Testing Methodology**
#### **Negation**
- Purpose: Test if SAC3 can handle cases where negation changes the meaning of a statement.
- Expected Behavior: SAC3 should detect inconsistencies if the model incorrectly interprets the negated questions as indicating that 29 is not a prime number.
- Example Test Cases:
  - *"The food is not bad."* → Should predict positive sentiment.

#### **Coreference Resolution**
- Purpose: Test if SAC3 can handle sentences where coreferences (e.g., pronouns or indirect references) are used, potentially altering the clarity of the question.
- Expected Behavior**: SAC3 should correctly maintain factual responses across all variations, detecting hallucinations if coreference misinterpretation causes inconsistent answers.
- Example Test Cases:
  - *"Did Albert Einstein develop relativity?"* → Variations with pronouns or indirect references.

#### **Temporal Reasoning**
- Purpose: Test if SAC3 correctly understands the order of events when temporal expressions are introduced.
- Expected Behavior: SAC3 should maintain consistency across all variations, confirming the correct sequence of events and flagging inconsistencies as hallucinations.
- Example Test Cases:
  - *"Did Obama serve as President before Trump?"*

#### **Semantic Role Labeling**
- Purpose: Test if SAC3 can handle variations where the roles of subjects and objects in a sentence are altered, potentially affecting meaning.
- Expected Behavior: SAC3 should consistently affirm that the dog chased the cat, detecting hallucinations if any variation leads to incorrect interpretations.
- Example Test Cases:
  - *"The dog chased the cat."* → Variations swapping subject and object.

#### **Fairness and Bias**
- Purpose: Evaluate SAC3’s sensitivity to questions that could reveal biases, such as associating certain attributes with specific groups.
- Expected Behavior: SAC3 should maintain consistency in factual information, detecting any hallucinations or biases if the model exhibits unexpected variations in responses based on gender, ethnicity, or nationality.
- Example Test Cases:
  - *"Is Einstein considered one of the greatest scientists?"*

#### **Robustness (Typographical Errors)**
- Purpose: Test SAC3’s robustness by introducing minor typographical errors in the question to see if it still provides consistent answers.
- Expected Behavior: SAC3 should handle these minor errors gracefully and maintain factual consistency. If SAC3 fails to do so, it should detect and report inconsistencies as hallucinations.
- Example Test Cases:
  - *"What is the capittal of France?"*

#### **Vocabulary and Synonym Substitution**
- Purpose: Check if SAC3 can handle synonymous or taxonomically related terms without hallucinating incorrect information.
- Expected Behavior: SAC3 should provide consistent responses across these variations, affirming that a sparrow is a bird without hallucinations.
- Example Test Cases:
  - *"Is a sparrow a bird?"* → *"Is a sparrow an avian?"*

---

### **4. Results**
#### **Behavioral Test Results**

| **Test Name**               | **Purpose**                                                                                     | **Expected Behavior**                                                                                                 | **Inference/Results**                                                                                                                                                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Negation**                | Tests if SAC3 handles cases where **negation** changes the meaning of a statement.              | SAC3 should detect inconsistencies when the model misinterprets negated questions (e.g., "29 is not a prime number"). | The results indicate that SAC3 was able to detect inconsistencies effectively for simple negation but struggled when negation was combined with complex sentence structures, as seen in the consistency score.                           |
| **Coreference Resolution**  | Tests if SAC3 resolves **pronouns** or **indirect references** in sentences effectively.         | SAC3 should correctly interpret coreferences and detect hallucinations if references are misinterpreted.              | SAC3 failed to resolve pronouns in certain cases, leading to incorrect interpretations of coreferences. The consistency votes reflected significant inconsistencies in pronoun-based perturbations.                                        |
| **Temporal Reasoning**      | Tests if SAC3 understands the **order of events** using temporal expressions.                   | SAC3 should identify and maintain the correct sequence of events, flagging inconsistencies as hallucinations.         | SAC3 showed limited capability in understanding temporal reasoning, particularly in distinguishing "before" and "after." The consistency scores revealed frequent failure in temporal perturbations.                                       |
| **Semantic Role Labeling**  | Tests SAC3's ability to handle variations in **subject-object roles** in sentences.             | SAC3 should affirm the correct roles in a sentence and detect hallucinations in misinterpretations.                   | The model struggled with subject-object role swaps and alternative phrasing, with frequent inconsistencies when detecting the agent or object in sentences. Consistency scores were low for these tests.                                 |
| **Fairness and Bias**       | Evaluates SAC3's sensitivity to biases in questions (e.g., gender, ethnicity, regional bias).    | SAC3 should maintain consistency and detect any unexpected variations based on biases in the prompts.                 | SAC3 exhibited some bias in responses, particularly when introducing demographic variations, such as gender and regional attributes. Although responses were consistent in many cases, subtle biases were evident in the consistency votes. |
| **Robustness (Typos)**      | Tests SAC3's robustness to **typographical errors** in questions.                               | SAC3 should handle minor typos gracefully, maintaining factual consistency.                                           | SAC3 handled simple typographical errors with high consistency; however, it faltered when multiple errors were introduced in the same sentence.                                                                                           |
| **Vocabulary Substitution** | Tests SAC3's handling of **synonyms** or **taxonomically related terms** in questions.          | SAC3 should affirm correct responses to synonymous variations without hallucinating.                                  | SAC3 performed well on basic synonym substitutions but showed inconsistencies with more nuanced or context-specific taxonomic terms.                                                                                                     |

### Notes
- Each test focuses on different aspects of SAC3's robustness, consistency, and accuracy.
- These tests highlight key challenges in detecting hallucinations and ensuring robustness in language models.



#### **Hallucination Detection Results**
- #### **Generated Tables**
    - Tables 2, 3, 4, and 5 inspired by the SAC3 paper:
        - #### 1. **Table 2: Accuracy for Hallucination Detection in Classification QA Tasks**
            - Fully replicated results from **classification QA tasks** using the Prime and Senator datasets.
            - ![image.png](attachment:598e31ce-aa5e-4d33-8ab0-ec1745ed74a4.png)
        - #### 2. **Table 3: Results for Unbalanced Datasets (100% Hallucinated Samples)**
            - Replicated only the **GPT-3 results** due to computational cost and resource constraints.
            - Results involving Falcon-7B and Guanaco-33b were not replicated.
            - ![image.png](attachment:23e61eb7-79d4-42f1-87d2-ed5c5a94a744.png)
    
        - #### 3. **Table 4: Results for Open-Domain Generation QA Tasks**
            - Not replicated due to the high computational cost and memory requirements.
    
        - #### 4. **Table 5: Impact of Thresholds and Model Types on Performance**
            - Not replicated due to limited resources and inability to evaluate multiple model

---


### **5. Challenges**

- ### **Resource Challenges** 
    ### 8 NVIDIA V100 32G GPUs
    - **NVIDIA V100**: A high-performance GPU designed for AI, deep learning, and data science tasks.
    - **32G**: 32 GB of memory per GPU, enabling the handling of large models and datasets.
    - **8 GPUs**: Indicates a cluster of 8 V100 GPUs working together, offering massive computational power through parallel processing.
    
    ### Comparison Table: V100 vs T4 GPUs
    | Feature                       | **NVIDIA V100 (32G)**               | **NVIDIA T4 (16G)**                   |
    |-------------------------------|--------------------------------------|---------------------------------------|
    | **Memory (per GPU)**          | 32 GB                               | 16 GB                                |
    | **TFLOPS (Tensor Performance)**| ~125 (FP16)                        | ~8.1 (FP16)                          |
    | **Architecture**              | Volta (optimized for large models)  | Turing (optimized for inference)     |
    | **Energy Consumption**        | 250 Watts                          | 70 Watts                             |
    | **Use Case**                  | Training large-scale models         | Inferencing and light training       |
    | **Relative Performance**      | ~10x faster for training tasks      | Optimized for cost-efficiency        |
    
    ### Performance Comparison: 8 V100s vs 2 T4s
    | Metric                        | **8 V100 32G GPUs**                 | **T4 x 2 GPUs**                      |
    |-------------------------------|--------------------------------------|---------------------------------------|
    | **Total Memory**              | 256 GB (8 × 32 GB)                 | 32 GB (2 × 16 GB)                    |
    | **Computational Power**       | ~1000 TFLOPS (Tensor Ops)          | ~16.2 TFLOPS (Tensor Ops)            |
    | **Relative Speed**            | ~60x faster for training           | Suitable for smaller models          |
    | **Suitability**               | Training large-scale LMs (e.g., Falcon-7B, GPT-3.5) | Inferencing or small workloads       |
    
    ### Challenges and Limitations
    1. **Reproducibility of Experiments from the Paper**:
       - Experiments such as **Table 3, Table 4, and Table 5** require substantial computational power and memory, making them only feasible on high-performance setups like **8 NVIDIA V100 GPUs**.
       - Attempting to replicate these experiments on smaller-scale hardware (e.g., T4 GPUs) will lead to significantly slower processing times and may not support the full dataset.
    
    2. **Impact of Limited Resources**:
       - Conducting tests on smaller datasets or with reduced perturbations (as feasible with free-tier GPUs) may result in deviations from the reported findings.
       - The reduced dataset size or hardware limitations mean the accuracy and AUROC metrics may not be fully reliable compared to those achieved with the original setup.
    
    3. **Resource Constraints on Free Platforms**:
       - Free-tier platforms like Google Colab and Kaggle do not provide sufficient GPU memory (e.g., typically T4 GPUs with 16 GB memory) to replicate experiments at scale.
       - Full-scale reproduction of SAC3 results demands high-performance GPUs, such as NVIDIA V100 or A100 clusters.
    
    ### Recommendations
    - To replicate the paper's findings meaningfully:
      - Invest in high-performance GPUs (e.g., V100 or A100 clusters).
      - Scale down the dataset size and perturbations for initial testing, keeping in mind the limitations of reduced computational power.
      - Acknowledge potential deviations in accuracy and consistency due to hardware constraints.

---


### **6. Conclusion**
- After conducting all the tests, I have concluded that the SAC3 methodology is impactful; however, the paper does not address certain tests covered in the checklist paper. I believe that combining the approaches from both papers would be more effective in detecting hallucinations in models.


---

### **7. References**
- Cite the relevant papers:
  - **CheckList**: Marco Tulio Ribeiro et al., ACL 2020. [Link to Paper](https://www.aclweb.org/anthology/2020.acl-main.442/)
  - **SAC3**: Jiaxin Zhang et al., EMNLP 2023. [Link to Paper](https://aclanthology.org/2023.findings-emnlp.1032/)


---







