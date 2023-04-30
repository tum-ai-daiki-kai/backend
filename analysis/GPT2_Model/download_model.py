from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_and_save_model(model_name: str, save_directory: str) -> None:
    """
    Download the pre-trained model and tokenizer, and save them locally.

    :param model_name: Name of the pre-trained model to download.
    :param save_directory: The directory to save the downloaded model and tokenizer.
    :return: None
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

if __name__ == "__main__":
    model_name = "gpt2-medium"
    save_directory = "gpt2-medium"
    download_and_save_model(model_name, save_directory)
