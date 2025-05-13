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

system_prompt = """Your task is to generate the clear and concise step by step solutions of the provided 3rd grade Math problem. Please consider the below instructions in your generation.

- You will be provided with the final answer, and additional Chinese explanation of the solution. When generating the step by step solution, you can leverage those information pieces, but you can also use your own judgment.  
- It is important that your generated step by step solution should be understandable as stand-alone, meaning that the student should not need to additionally check final answer or explanation provided.
- Your solution steps will be later used to identify the knowledge concepts associated at each step. Therefore, please don't write a final conclusion sentence as the last step, because it won't contribute to any knowledge concept.
- Don't generate any text other than the step by step solution described earlier.
- Don't divide any equation to multiple lines, i.e. an equation should start and finish at the same line. 
- Make your step-by-step solution concise (e.g. not much verbose, and not a longer list than necessary) as described earlier.
- You must provide your step by step solution in a structured and concise manner in Solution_Steps field as a list of steps, i.e. [<step1>, ..., <stepN>] . Don't enumerate the steps.
- You have limited tokens, try to make each <step> as concise as possible. 
- IMPORTANT: If your final answer does not match the provided final answer, it's fine. DON'T try to start a new solution! and DON'T even write that the answers don't match. The reason is, most of the times your errors are just basic calculation errors which we can tolerate anyways. 
- IMPORTANT: Don't use any invalid character as I will later call ast.literal_eval on your response message.

Please follow the example output in json format below as a template when structuring your output.

{"Solution_Steps": <Structured solution steps as a list, i.e., [<step1>, ..., <stepN>] .>}"""

user_prompt_template="""Question: {}
Final Answer: {}
Explanation: {}
"""

def structure_answer(item):
    """Function for handling the answer structure of different types of questions.
    If the question is fill-in-the-blank, 填空, it will return a single string or comma separated strings.
    If the question is multiple choice, 单选, it will return a choice letter (e.g. A) and the corresponding text, separated by : .

    Args:
        item (dict): one problem element from the json file. 
    """
    
    #If fill-in-the-blank
    if item["type"] == "填空":
        #If there are multiple blanks
        if len(item["answer"]) > 1:
            return ", ".join(item["answer"])
        else:
            return item["answer"][0]
    #If multiple choice
    elif item["type"] == "单选":
        choice = item["answer"][0]
        return f"{choice}: {item['options'][choice]}"
    else:
        raise ValueError(f"Unknown question type: {item['type']}")
    
def create_full_user_prompt(item):
    """Function for creating the full user prompt for a given problem item.

    Args:
        item (dict): one problem element from the json file. 
    """
    answer_structured = structure_answer(item)
    return user_prompt_template.format(item['question'], answer_structured, item["analysis"])

def get_soluton_steps(item):
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
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content  # Assuming this returns the converted text

# Load JSON data
with open(annotated_question_file, 'r', encoding='utf-8') as file:
    data_written = json.load(file)

# Load JSON data
with open(original_question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

counter = 0

# Iterate and convert each problem
for key, item in data.items():

    # Check if it is already annotated
    flag = True
    if "step_by_step_solution_gpt4o" in data_written[key].keys():
        if len(data_written[key]["step_by_step_solution_gpt4o"]) != 0:
            flag = False
            print("The following key is already annotated => ", key)

    if flag:

        start_time = time.time()  # Capture the start time

        try: 
            response = get_soluton_steps(item)  # Add the converted question to the dictionary

            item['step_by_step_solution_gpt4o'] = ast.literal_eval(response)

        except:
            item['step_by_step_solution_gpt4o'] = {}
            print("Eror with key ", key)
        
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