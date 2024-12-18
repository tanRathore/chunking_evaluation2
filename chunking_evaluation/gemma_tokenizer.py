import sentencepiece as spm
import os

# Specify the directory path where all your corpus files are stored
corpus_dir = "/Users/tanishqsingh/Documents/GitHub/chunking_evaluation/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/"

# List all files in the directory (ensure they are text files)
file_list = [os.path.join(corpus_dir, file) for file in os.listdir(corpus_dir) if file.endswith('.md')]

# Combine all file paths into a single string separated by commas
corpus_files = ",".join(file_list)

# Model prefix and other settings
model_prefix = "gemma_tokenizer"

# Train the SentencePiece model
if file_list:  # Check if files were found
    spm.SentencePieceTrainer.train(
        input=corpus_files,
        model_prefix=model_prefix,
        vocab_size=7822,
        character_coverage=0.9995,
        model_type='bpe'
    )
    print("SentencePiece model trained successfully!")
else:
    print("No text files found in the specified directory.")
