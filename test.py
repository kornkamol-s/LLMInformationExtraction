import pandas as pd
import json
import ast

# Load JSONL file into a DataFrame
jsonl_file = 'data/training/data_partitioning/test/project_description_test_answer.jsonl'  # Replace with your file path
data = []
data = []
with open(jsonl_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Define a function to process the 'value' field
def process_value(value):
    # If value is a dictionary or list, convert it to a string representation
    if isinstance(value, (dict, list)):
        return json.dumps(value)

    # If value is a string that looks like a dictionary, parse it and reformat
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, (dict, list)):
            return json.dumps(parsed_value)
    except (ValueError, SyntaxError):
        pass

    # Return the value as-is if it's a basic string or unrecognized structure
    return str(value)

# Apply the function to the 'value' column, storing everything in a new column called 'answers'
df['answers'] = df['value'].apply(process_value)

# Select only the 'answers' column and save to CSV
df[['answers']].to_csv('output.csv', index=False)
print("Data successfully saved to output.csv")