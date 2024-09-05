# Fine-Tuning Llama 3.1 8B Instruct for SQL Generation using QLORA
![image](https://github.com/user-attachments/assets/be967db7-9298-43cf-b9a9-7452fb6fea45)

Kaggle Notebook: https://www.kaggle.com/code/bertpro/finetuning-llama-3-1-8b-transformers-library (I ran the project on two T4 15 GB GPUS available on Kaggle)

This project demonstrates how to fine-tune the Meta Llama 3.1 8B Instruct model for natural language to SQL query translation using the QLORA (Quantized Low-Rank Adaptation) technique.  The project focuses on achieving efficient fine-tuning with reduced memory usage and faster training times.

## Overview

The project utilizes the Hugging Face `transformers` and `peft` libraries to:

- Load and preprocess the Spider dataset, a benchmark for natural language to SQL translation.
- Apply 4-bit quantization to the Llama 3.1 model to reduce its memory footprint.
- Implement LoRA (Low-Rank Adaptation) to fine-tune only a small subset of the model's parameters, further enhancing efficiency.
- Train the model using the `Seq2SeqTrainer` from `transformers`, which is specifically designed for sequence-to-sequence tasks.
- Save the fine-tuned model with merged LoRA weights for easy inference.
- Showcase inference examples with streaming output to demonstrate the model's capabilities.

## Requirements

- Python 3.8 or higher
- PyTorch 1.12 or higher
- transformers library (`pip install transformers`)
- datasets library (`pip install datasets`)
- peft library (`pip install peft`)
- accelerate library (`pip install accelerate`)
- bitsandbytes library (`pip install bitsandbytes`)
- evaluate library (`pip install evaluate`)
- Tensorflow 2.11 or higher (`pip install tensorflow`)

- **GPU with Sufficient Memory:** A GPU with at least 16GB of memory is recommended for running this project.  You can use cloud-based GPU services like Google Colab or Kaggle. 

## Usage

1. **Setup:**
   - Install the required libraries listed above.
   - Set the `HUGGING_FACE_HUB_TOKEN` environment variable to your Hugging Face API token.  You can get a token from your Hugging Face account settings. 
   - Download the Spider dataset using the `datasets` library: `spider_dataset = load_dataset("spider")`.

2. **Preprocessing and Tokenization:**
   - Randomly sample a subset of the training data (adjust the `num_samples` variable if needed).
   - Tokenize the questions and SQL queries using the Llama 3.1 tokenizer.
   - Pad the sequences to a fixed `max_length` determined by analyzing the sequence length distribution. 

3. **QLORA Configuration and Model Loading:**
   - Define the LoRA configuration using `LoraConfig`. Experiment with `target_modules` and other LoRA parameters for optimal performance and memory usage.
   - Load the pre-trained Llama 3.1 8B Instruct model in 4-bit precision using `BitsAndBytesConfig`.
   - Wrap the model with LoRA adapters using `get_peft_model` from the `peft` library.

4. **Training:**
   - Define the training arguments using `Seq2SeqTrainingArguments`. Adjust the batch size, gradient accumulation steps, learning rate, and other hyperparameters as needed.
   - Create a `DataCollatorForSeq2Seq` to handle data collation during training.
   - Create a `Seq2SeqTrainer` instance, passing the model, training arguments, datasets, and data collator.
   - Train the model using `trainer.train()`. 

5. **Save the Fine-Tuned Model:**
   - After training, merge the LoRA weights back into the base model using `model.merge_and_unload()`.
   - Save the merged model to a directory on your device using `model.save_pretrained()`. 
   - Also, save the tokenizer to the same directory using `tokenizer.save_pretrained()`.

6. **Inference:**
   - Load the fine-tuned, merged model using `AutoModelForCausalLM.from_pretrained()`.
   - Use the `model.generate()` function to generate SQL queries for input natural language questions.
   - Implement streaming output using `TextIteratorStreamer` to provide real-time feedback during generation. 

## Challenges and Considerations

- **Memory Management:** Training and even inferencing with large language models like Llama 3.1 require significant GPU memory. Carefully choose batch size, sequence length, and other parameters to avoid out-of-memory errors.
- **Evaluation:**  While this project focuses on demonstrating fine-tuning and inference, a comprehensive evaluation of the model's accuracy is essential. Consider implementing metrics like exact match accuracy and execution accuracy in a future iteration.
- **Data Quality and Diversity:** The quality and diversity of the training data are crucial for the model's performance.  Explore techniques for data augmentation and ensure the training data covers a wide range of SQL syntax and query structures. 
- **Model Choice:** The pre-trained Llama model, while powerful for natural language, might not be the optimal choice for SQL generation. Explore other models specifically designed for code generation. 

## Conclusion

This project provides a starting point for fine-tuning Llama 3.1 for SQL generation using QLORA. It highlights the importance of memory optimization techniques, the use of `peft` for efficient fine-tuning, and the application of appropriate decoding strategies for sequence-to-sequence tasks.  
