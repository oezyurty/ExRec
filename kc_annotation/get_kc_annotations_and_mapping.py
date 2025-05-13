import json
from openai import OpenAI
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--original_question_file', type=str, help='Path to the original question file')
parser.add_argument('--annotated_question_file', type=str, help='Path to the annotated question file')
args = parser.parse_args()

original_question_file = args.original_question_file
annotated_question_file = args.annotated_question_file

client = OpenAI()

system_prompt = """You will be provided with a Math question and its step by step solution. Your task is to provide the concise and comprehensive list of knowledge concepts (KCs) in Math curriculum required to correctly answer the questions. Then, you will map each solution step with its associated KC(s).

Your task has mainly 3 phases, whose details will be provided below. Each phase has its own field in your json output format. 
- Reasoning: 
    1. Identify all the relevant KCs required to solve this problem.
    2. Justify why each KC is relevant, considering the question and solution steps.
    3. You have limited space, so please use 100 words maximum. 

- List of KCs: Provide a **list of unique KCs** with the help of your reasoning above, i.e. [<KC 1>, ..., <KC M>] . Don't enumerate the KCs.
    1. Provide multiple knowledge concepts only when it is actually needed.
    2. Some questions require a figure, which you won't be provided. As the step-by-step solution is already provided, Use your judgement to infer which knowledge concept(s) might be needed.
    3. For a small set of solutions, their last step(s) might be missing due to limited token size. Use your judgement based on your input and your ability to infer how the solution would conclude.
    4. Remember that knowledge concepts should be appropriate for Math curriculum. If annotated step-by-step solution involves advanced techniques, use your judgment for more simplified alternatives.

- Mapping between solution steps and KCs: All solution steps and all knowledge concepts must be mapped, while many-to-many mapping is indeed possible. 
    IMPORTANT: Each solution step is already numbered from 1 to N. Here, also assume that each knowledge concept is numbered from 1 to M, where M is the number of KCs you found earlier. For consistency, use the same ordering as your output of list of KCs. Your output should enumerate all solution step - knowledge concept pairs as numbers. 
    1. Each solution step has to be paired. 
    2. Each knowledge concept has to be paired.
    3. Map a solution step with a knowledge concept only if they are relevant.
    4. Your pairs cannot contain artificial solution steps. For instance, If there are 4 solution steps, the pair "5-2" is indeed illegal.
    5. Your pairs cannot contain artificial knowledge concepts. For instance, If there are 3 knowledge concepts, the pair "3-5" is indeed illegal.
    IMPORTANT: For this field, you will output solution step - knowledge concept pairs in a comma separated manner and in a single line. For example, if there are 4 solution steps and 5 knowledge concepts, one potential output could be the following: "1-1, 1-3, 1-5, 2-4, 3-2, 3-5, 4-2, 4-3, 4-5". The provided example format is only for clarity. The output should be specific to the given question and solution, and not follow the exact structure or content of any example.

IMPORTANT NOTE: For your task, try to use the Common Core State Standards for Mathematics for the Knowledge Concept (KC) annotations. The reason is, we aim to get consistent texts for the same KCs across different questions. 

Please follow the example output in json format below as a template when structuring your output. IMPORTANT: Don't use any invalid character as I will later call ast.literal_eval on your response message.

{"Reasoning": <Your reasoning to identify relevant KCs.>,
"list_KCs": <list of unique KCs, i.e. [<KC 1>, ..., <KC M>] .>},
"mapping_step_KC": <solution step - knowledge concept pairs in a comma separated manner and in a single line.>}"""

user_prompt_template="""Question: {}

Solution steps: {}"""

def get_structured_sol_steps(item):
    """Function for structuring solution steps for a given problem item.

    Args:
        item (dict): one problem element from the json file. 

    Returns:
        str: structured solution steps.
    """
    sol_steps = item["step_by_step_solution_gpt4o"]["Solution_Steps"]
    structured_sol_steps = ""
    for i, step in enumerate(sol_steps):
        structured_sol_steps += f"{i+1}) {step}\n"
    return structured_sol_steps

def create_full_user_prompt(item):
    """Function for creating the full user prompt for a given problem item.

    Args:
        item (dict): one problem element from the json file. 
    """
    solution_steps = get_structured_sol_steps(item)
    return user_prompt_template.format(item['question'], solution_steps)

def get_mapping(item):
    # Get the full user prompt 
    full_user_prompt = create_full_user_prompt(item)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                    "text": system_prompt,
                    "type": "text"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": full_user_prompt
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    return response.choices[0].message.content  # Assuming this returns the converted text

# Load JSON data
with open(original_question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

counter = 0
max_retries = 3 

# Iterate and convert each problem
for key, item in data.items():

    # Check if it is already annotated
    flag = True
    if "kc_mapping_gpt4o" in data[key].keys():
        if len(data[key]["kc_mapping_gpt4o"]) != 0:
            flag = False
            print("The following key is already annotated => ", key)

    if flag:

        start_time = time.time()  # Capture the start time

        for attempt in range(max_retries):
            try:
                response = get_mapping(item)  # Attempt to get the mapping
                item['kc_mapping_gpt4o'] = ast.literal_eval(response)
                break  # Exit the loop if successful
            except Exception as e:
                print(f"ERROR on attempt {attempt + 1} with key {key}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Optional: wait before retrying
                else:
                    item['kc_mapping_gpt4o'] = {}  # Handle the error after all retries
                    print("Giving up after 3 attempts")
        
        end_time = time.time()  # Capture the end time
        iter_time = end_time - start_time  # Calculate the time taken for this iteration
        
        print(f"The question {key} took {iter_time:.2f} seconds to convert")
        
        counter += 1  # Increment the counter

        # Save the progress at every 100 iterations
        if counter % 20 == 0:
            with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
                json.dump(data, temp_file, ensure_ascii=False, indent=2)
            print(f"Progress saved at iteration {counter}")

with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(data, temp_file, ensure_ascii=False, indent=2)
print(f"Progress saved at iteration {counter}")