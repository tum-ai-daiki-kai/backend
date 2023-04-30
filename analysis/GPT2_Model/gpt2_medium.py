#%%
import logging
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from typing import Tuple
from datasets import Dataset
import json

logging.basicConfig(level=logging.INFO)

# %%
def load_csv_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    :param file_path: The path to the CSV file.
    :return: A DataFrame containing the dataset.
    """
    logging.info(f"Loading dataset from {file_path}")
    dataset = pd.read_csv(file_path)
    logging.info(f"Dataset loaded with {dataset.shape[0]} rows and {dataset.shape[1]} columns")
    return dataset

def tokenize_data(tokenizer, dataset: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Tokenize input and output text in the dataset using the given tokenizer.

    :param tokenizer: The tokenizer to use for tokenizing the text.
    :param dataset: The dataset to tokenize.
    :return: A tuple containing tokenized input and output text.
    """
    logging.info("Tokenizing input and output text")
    tokenized_input = dataset['Input'].apply(lambda x: tokenizer.encode(x, truncation=True))
    tokenized_output = dataset['Output'].apply(lambda x: tokenizer.encode(x, truncation=True))
    logging.info("Tokenization complete")
    return tokenized_input, tokenized_output

def prepare_dataset(tokenized_input: pd.Series, tokenized_output: pd.Series) -> Dataset:
    """
    Prepare the dataset for training.

    :param tokenized_input: Tokenized input text.
    :param tokenized_output: Tokenized output text.
    :return: A Dataset object for training.
    """
    logging.info("Preparing dataset for training")
    examples = [tokenizer.build_inputs_with_special_tokens(inp, out) for inp, out in zip(tokenized_input, tokenized_output)]
    dataset = Dataset.from_dict({'input_ids': examples})
    logging.info("Dataset preparation complete")
    return dataset

def fine_tune_model(model, dataset, output_dir: str, epochs: int = 20) -> None:
    """
    Fine-tune the GPT-2 model using the given dataset.

    :param model: The GPT-2 model to fine-tune.
    :param dataset: The dataset to use for fine-tuning.
    :param output_dir: The directory to save the fine-tuned model.
    :param epochs: The number of epochs to train the model.
    :return: None
    """
    logging.info("Fine-tuning the model")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=6,
        save_steps=20_000,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    logging.info("Starting training")
    trainer.train()
    logging.info("Training complete")

def generate_text(model, tokenizer, prompt: str, max_length: int = 200, min_length: int = 100) -> str:
    """
    Generate text using the fine-tuned model given a prompt.

    :param model: The fine-tuned GPT-2 model.
    :param tokenizer: The tokenizer used with the model.
    :param prompt: The text prompt to generate text from.
    :param max_length: The maximum length of the generated text.
    :return: The generated text.
    """
    logging.info(f"Generating text with prompt: {prompt}")

    # Set the padding token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    encoding = tokenizer(prompt, return_tensors='pt', return_attention_mask=True, padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Adjust the temperature and set no_repeat_ngram_size
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        num_return_sequences=4,
        top_k=30,
        top_p=0.92,
        temperature=0.6,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
    )

    generated_text = tokenizer.decode(output[0])
    return generated_text



#%%
##### ANALYSIS #####
dataset_path = "finetuning_data.csv"
dataset = load_csv_dataset(dataset_path)

#%%
dataset.head()

#%%
# Initialize the tokenizer and the model
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#%%
# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token
# Tokenize the dataset and prepare it for training
tokenized_input, tokenized_output = tokenize_data(tokenizer, dataset)
training_dataset = prepare_dataset(tokenized_input, tokenized_output)

#%%
# Fine-tune the model
output_dir = "fine_tuned_gpt2_medium"
fine_tune_model(model, training_dataset, output_dir, epochs=4)

#%%
# Save the fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

#%%
# Load the fine-tuned model and generate text
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

#%%
# generate text with finetuned model
prompt = "Create a short movie teaser including a description of the main character for an Drama movie."
generated_text = generate_text(fine_tuned_model, fine_tuned_tokenizer, prompt)
print(generated_text)


#%%
filename = "../genres.csv"
genres_df = pd.read_csv(filename)
genres_df.head()

# %%
columns_to_retain = [
    "Genre",
]
filtered_df = genres_df[columns_to_retain]
filtered_df.head()

# %%
# Create an empty DataFrame to store the 'Genre' and 'Movie Teaser' columns
new_df = pd.DataFrame(columns=['Genre', 'Movie Teaser'])

#%%
# Create two empty lists to store the 'Genre' and 'Movie Teaser' values
genres = []
movie_teasers = []

# Iterate over the values in the 'column_name' column of the filtered_df DataFrame
for value in filtered_df['Genre']:
    for i in range(10):
        prompt = f"Create a short movie teaser including a description of the main character for an {value} movie."
        generated_text = generate_text(fine_tuned_model, fine_tuned_tokenizer, prompt)

        # Append the 'Genre' and 'Movie Teaser' values to the respective lists
        genres.append(value)
        movie_teasers.append(generated_text)

# Combine the 'Genre' and 'Movie Teaser' lists into a new DataFrame
new_df = pd.DataFrame({'Genre': genres, 'Movie Teaser': movie_teasers})

# Display the new DataFrame with the 'Genre' and 'Movie Teaser' columns
print(new_df.head())

# %%
# Save the DataFrame as a CSV file
output_filename = "analysis_data.csv"
new_df.to_csv(output_filename, index=False)