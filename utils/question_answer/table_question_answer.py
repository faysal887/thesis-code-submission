# https://stackoverflow.com/questions/77102807/error-while-using-tapas-pipeline-for-table-based-question-answering

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd


# Define the TAPAS model and tokenizer
model_name = "google/tapas-base-finetuned-wtq"
model = TapasForQuestionAnswering.from_pretrained(model_name, cache_dir='/data/faysal/thesis/models/huggingface')
tokenizer = TapasTokenizer.from_pretrained(model_name, cache_dir='/data/faysal/thesis/models/huggingface')

def ask_model(table, query):
    # since in the demo code, which I found, the model takes queries as a list, so passing single query in a brackets
    queries=[query]
    # Tokenize the table and queries
    inputs = tokenizer(        
                table=table, 
                queries=queries,
                padding="max_length", 
                return_tensors="pt", 
                truncation=True
            )


    # Pass the tokenized inputs to the TAPAS model to obtain predictions
    outputs = model(**inputs)


    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )

    # Map aggregation indices to human-readable aggregation types
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

    # Extract and format the answers based on the predicted answer coordinates and aggregation types
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # Only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # Multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(str(table.iat[coordinate]))
            answers.append(", ".join(cell_values))


    for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
        if len(queries)==1:
            if answer:
                return answer
            else:
                answer='sorry, I cannot find the answer'
                return answer
        else:
            return 'invalid input'