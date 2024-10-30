from datasets import load_dataset
import pandas as pd


def _transform_record(row):
    return {
                "messages": [
            {
                "role": "system",
                "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If no relevant information is found, do not return anything."
            },
            {
                "role": "user",
                "content": f"Question: {row['question']}\n\nContext: {row['context']}"
            },
            {
                "role": "assistant",
                "content": f"{row['answers']}"
            }
        ]
    }

def _transform_record_test(row):
    return {
                "messages": [
            {
                "role": "system",
                "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If no relevant information is found, do not return anything."
            },
            {
                "role": "user",
                "content": f"Question: {row['question']}\n\nContext: {row['context']}"
            }
        ]
    }


def remove_duplicate_answers(answers):
    return list(set(answers))


# def _transform_record_test(row):
#     return {
#                 "input": f"Question: {row['question']}\n\nContext: {row['context']}",
#                 "output": f"{row['answers']}",
#                 "answer": f"{row['model_answers']}"
#             }

dataset = load_dataset('rajpurkar/squad')

training_data = dataset['train'].remove_columns(['id', 'title']) 
df = pd.DataFrame(dataset['train'])
df['answers'] = df['answers'].apply(lambda x: x['text'])
df = df.apply(lambda row: _transform_record(row), axis=1)
df.head(10000).to_json('data/intermediate/squad/squad_train.jsonl', orient='records', lines=True)

validation_data = dataset['validation'].remove_columns(['id', 'title']) 

df = pd.DataFrame(dataset['validation'])
df['answers'] = df['answers'].apply(lambda x: x['text'])
df['answers'] = df['answers'].apply(remove_duplicate_answers)

val_df = df.sample(frac=0.5, random_state=42)
val_df = val_df.apply(lambda row: _transform_record(row), axis=1)
val_df.head(2000).to_json('data/intermediate/squad/squad_validate.jsonl', orient='records', lines=True)

test_df = df.drop(val_df.index)
# model_answer_df = pd.read_csv('../Downloads/answer-after-tune.csv')

# test_df = test_df.head(1000).reset_index(drop=True)
# test_df = pd.concat([test_df, model_answer_df], axis=1)

test_df[['answers']].head(2000).to_json('data/intermediate/squad/squad_test_answer.jsonl', orient='records', lines=True)
test_df = test_df.apply(lambda row: _transform_record_test(row), axis=1)

test_df.head(2000).to_json('data/intermediate/squad/squad_test_prompt.jsonl', orient='records', lines=True)
