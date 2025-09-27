# GPT-2 Fine-Tuning on WikiText-2

[![Hugging Face Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-blue.svg)](https://huggingface.co/docs/transformers/index)
[![Datasets](https://img.shields.io/badge/🤗%20Datasets-2.x-blue.svg)](https://huggingface.co/docs/datasets/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, well-documented example of fine-tuning the GPT-2 language model on the WikiText-2 dataset using the Hugging Face 🤗 Transformers library. This project is designed to run efficiently on a single GPU with 16GB of VRAM (e.g., NVIDIA T4).

This repository serves as a practical introduction to causal language modeling (CLM) and is an excellent starting point for understanding how to adapt pre-trained models to new domains.

## 🚀 Features

- **Full Pipeline**: From data loading and tokenization to training and inference.
- **Resource Efficient**: Configured to run on a single T4 GPU (16GB VRAM) with mixed precision (`fp16`).
- **Best Practices**: Includes proper dataset chunking, use of a data collator, and evaluation.
- **Ready-to-Use Code**: The code is self-contained and can be easily adapted to fine-tune on your own text corpus.
- **Inference Example**: Includes a simple script to test the fine-tuned model's text generation capabilities.

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.7+
- PyTorch (>=1.9.0, preferably with CUDA)
- An NVIDIA GPU with sufficient VRAM (e.g., T4, V100) is highly recommended.

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GPT2-WikiText-FineTuning.git
   cd GPT2-WikiText-FineTuning```
----
## Install the required Python packages
  ```bash
  pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start
1)Run the training script. The script will automatically download the WikiText-2 dataset and the gpt2 model.
  ```bash
  python train.py
```
This will:

-Download and preprocess the WikiText-2 dataset.

-Fine-tune the GPT-2 model for 1 epoch (configurable).

-Save the best model to ./gpt2-finetuned/.

-Run a quick text generation test at the end.

2)Test the fine-tuned model. After training, you can use the provided example or create your own.
  ```bash
  from transformers import pipeline

  generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="gpt2")
  print(generator("The theory of relativity states that", max_new_tokens=50))
```
## 📊 Results
After fine-tuning for just 1 epoch, the model should start generating text that is more consistent with the style and vocabulary of Wikipedia articles. The validation loss should decrease, indicating the model is learning the patterns in the WikiText-2 dataset.

Example input:

"Deep learning is"

Example output from the fine-tuned model might look like:

"Deep learning is a branch of machine learning based on artificial neural networks. It has been applied to fields such as computer vision, speech recognition, and natural language processing..."

## 🛠️ Code Overview
-train.py: The main script containing the entire fine-tuning pipeline.

### Key Steps:

-Data Loading: Uses the datasets library to load wikitext-2-raw-v1.

-Tokenization: Uses the GPT-2 tokenizer. The pad token is set to the EOS token.

-Chunking: Groups the tokenized texts into blocks of 128 tokens for efficient training.

-Training: Uses the Trainer API with TrainingArguments optimized for a T4 GPU (e.g., fp16=True, per_device_batch_size=2).

-aving & Inference: The model is saved and then tested with a text-generation pipeline.

## 📈 Future Improvements / Ideas
### This project is a starting point. Here are some ways you can extend it:

-Hyperparameter Tuning: Experiment with learning_rate, num_train_epochs (3+), and block_size (512).

-Larger Model: Try fine-tuning larger variants like gpt2-medium or gpt2-large on a more powerful GPU.

-Different Dataset: Fine-tune on a custom dataset (e.g., your own blog posts, code, or song lyrics).

-Logging: Integrate with Weights & Biases or TensorBoard for better experiment tracking.
## 🤝 Contributing
Contributions are welcome! If you have ideas for improvements, please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

