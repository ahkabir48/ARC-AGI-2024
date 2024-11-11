import os
from cerebras.cloud.sdk import Cerebras
import arckit
import numpy as np

train_set, eval_set = arckit.load_data()



def call_prompt(prompt_file, arr1, arr2=None, conversation_history=[]):
    # Set the Cerebras API key
    api_key = "API-KEY"
    model_name = "llama3.1-8b"
    os.environ["CEREBRAS_API_KEY"] = api_key
    
    # Initialize the Cerebras client
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
    
    # Read and format the prompt template
    with open(prompt_file, 'r') as f:
        text_prompt_template = f.read()
    if not (arr2 is None): 
        text_prompt = text_prompt_template.format(
        array1=arr1, 
        array2=arr2)
    else:
        text_prompt = text_prompt_template.format(
        array1=arr1)
    
    # Create the chat completion request
    chat_completion = client.chat.completions.create(
        messages=conversation_history+[
            {
                "role": "user",
                "content": text_prompt,
            }
        ],
        model=model_name,
    )
    
    # Return the response content
    return chat_completion.choices[0].message.content


def conversation(arr1, arr2, arr3):
    conversation_history = []

    response1 = call_prompt('prompts/prompt1.txt', arr1, arr2)
    conversation_history.append({"role": "assistant", "content": response1})

    response2 = call_prompt('prompts/prompt2.txt', arr1, arr2)
    conversation_history.append({"role": "assistant", "content": response2})
    response3 = call_prompt('prompts/prompt3.txt', arr1, arr2)
    conversation_history.append({"role": "assistant", "content": response3})
    response4 = call_prompt('prompts/prompt4.txt', arr3)

    return response4

f = open("submission.csv", 'w')
f.write("output_id,output\n")

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred
import re

def parse_to_numpy_array(s):
    # Step 1: Clean up unwanted characters, replace ",," with a single comma, remove extra brackets
    s = s.replace("[", "").replace("]", "").replace(",,", ",").replace(",", " ")
    
    # Step 2: Split into rows based on newline
    rows = s.strip().split("\n")
    
    # Step 3: Convert each row into a list of integers
    array = [list(map(int, re.findall(r'\d+', row))) for row in rows]
    
    # Step 4: Convert the list of lists to a numpy array
    return array

def call(batch, i):
    response = conversation(
        arr1 = batch.train[0][0],
        arr2 = batch.train[0][1],
        arr3 = batch.test[i][0]
    )
    return batch.id + '_' + str(i) +',' + flattener(parse_to_numpy_array(response)) + '\n'

for batch in eval_set:
    if type(batch.train[0][0]) != np.ndarray:
        f.write(batch.id + '_0,||' + '\n')
        f.write(batch.id + '_1,||' + '\n')
        continue 
    l = int(len(batch.test))
    for i in range(l):
        try:
            r = call(batch, i)
            f.write(r)
            print('success')
        except:
            try:
                call(batch, i)
                f.write(r)
                print('success')
            except:
                print('failed')
                f.write(batch.id + '_'+str(i)+',||' + '\n')
    





f.close()
print(eval_set.score_submission(
    'submission.csv', # Submission with two columns output_id,output in Kaggle fomrat
    topn=3,           # How many predictions to consider (default: 3)
    return_correct=False # Whether to return a list of which tasks were solved
    ))
print('complete')
# id = 20
# testnum = 1
# pred = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 4, 0, 1, 0, 0, 0, 0],
#         [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
#         [0, 4, 0, 1, 0, 4, 0, 0, 0, 0],
#         [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
#         [0, 1, 0, 4, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# def flattener(pred):
#     str_pred = str([row for row in pred])
#     str_pred = str_pred.replace(', ', '')
#     str_pred = str_pred.replace('[[', '|')
#     str_pred = str_pred.replace('][', '|')
#     str_pred = str_pred.replace(']]', '|')
#     return str_pred

# f = open("submission_test.csv", "a")
# f.write(str(id) + '_' + str(testnum) + ", " + flattener(pred))
# f.close