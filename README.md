# Dell AI Delivery Academy - Deep Learning and Generative AI
### [Dr. Elias Jacob de Menezes Neto](https://docente.ufrn.br/elias.jacob)
Welcome to the Dell AI Delivery Academy course on Deep Learning and Generative AI. This repository contains the code and resources for the course. Our syllabus covers a wide range of topics in these fields. Yeah, it is a lot, I know. But we will have fun!

> This course will be taught between 2024-08-14 and 2024-09-04 at [Instituto Metrópole Digital/UFRN](https://portal.imd.ufrn.br/portal/)

## Deep Learning

In the Deep Learning section, we will explore the following topics:

- **Perceptron and Activation Functions**: Learn about the building blocks of neural networks and how they enable learning complex patterns.
- **Multilayer Perceptrons**: Understand how multiple layers of perceptrons can be combined to create powerful deep learning models.
- **Backpropagation, Stochastic Gradient Descent, and Optimizers**: Dive into the training process of neural networks, including the backpropagation algorithm, stochastic gradient descent, and various optimization techniques.
- **Basics of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs)**: Get an introduction to two popular architectures used in deep learning: CNNs for image and video processing, and RNNs for sequential data analysis.
- **Overview of Computer Vision**: Explore various tasks in computer vision, such as image segmentation, enhancement, and feature extraction.
- **Tools: PyTorch and TensorFlow/Keras**: Gain hands-on experience with popular deep learning frameworks. We'll focus on PyTorch.
- **Practical Exercises**: Apply your knowledge through practical exercises and projects.

## Generative AI

The Generative AI section will cover the following topics:

- **Transformer Architecture**: Learn about the revolutionary transformer architecture, including self-attention mechanisms and the encoder-decoder structure. Understand the impact of models like BERT.
- **Embeddings**: Dive into the concept of embeddings and their role in representing text, images, and other data types in generative models.
- **Large Language Models (LLMs)**: Explore the world of foundation models, fine-tuning techniques, and prompt engineering for generating human-like text.
- **Retrieval Augmented Generation (RAG)**: Understand how retrieval-based methods can enhance the performance and capabilities of generative models.
- **Tools: Hugging Face, PyTorch/TensorFlow, Langchain, and LlamaIndex**: Get hands-on experience with popular libraries and frameworks for building and deploying generative AI models.

Throughout the course, you will have the opportunity to work on practical exercises and projects to solidify your understanding of deep learning and generative AI concepts. I will provide code examples and resources to support your learning journey.

## Prerequisites

To get started with the course, ensure that you have the following prerequisites:

- Access to a machine with a GPU (recommended) or Google Colab for running computationally intensive tasks.
- Installation of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for managing Python environments.
- An account with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization.

## Installation

Follow these steps to set up the environment and dependencies:

1. **Download this Repository:**

   Clone this repository to your local machine using:

   ```shell
    git clone https://github.com/eliasjacob/dell_deep_learning_genai.git
    cd dell_deep_learning_genai
    ```

2. **Run the Download Script:**

   Run the following script to download the required resources:

   ```shell
   bash download_datasets_and_binaries.sh
   ```

3. **Install Ollama:**

   Download and install Ollama from [here](https://ollama.com/download).

4. **Download LLama 3.1:**

   After installing Ollama, run the following command to download LLama 3.1:

   ```bash
   ollama pull llama3.1
   ```

5. **Set up Conda Environment:**

   Create and activate the Conda environment using the provided `environment.yml` file. For better performance, I suggest you use [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) instead of Conda. It looks like Conda, but it is faster. Just replace `conda` with `mamba` in the commands below:

   ```bash
   conda env create -f environment.yml
   conda activate dell
   ```

6. **Authenticate Weights & Biases:**

   Create a Weights & Biases account if you haven't already. Then, authenticate by running:

   ```bash
   wandb login
   ```

## Getting Started

Once you have the environment set up, you can start exploring the course materials, running the code examples, and working on the practical exercises.

### Notes

- Certain parts of the code may require a GPU for efficient execution. If you don't have access to a GPU, you can use Google Colab for running the code.

## Contributing

Contributions to the course repository are welcome! If you have any improvements, bug fixes, or additional resources to share, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Create a Pull Request.

## Contact

You can [contact me](elias.jacob@ufrn.br) for any questions or feedback regarding the course materials or repository.


## Note

I've prepared the material for some topics that were not supposed to be covered in this course. Since I've prepared that for you, I'll leave it here in a folder called `extra`. If you want to take a look at it, feel free to do so. Here are the topics:

- **Overview of Reinforcement Learning**: Learn about agents, environments, and reward optimization in reinforcement learning.
- **Time Series Analysis**: Study techniques for decomposition, stationarity analysis, and autoregressive models in time series data.
- **Overview of Anomaly Detection**: Discover methods for identifying unusual patterns or outliers in data.
- **Overview of Natural Language Processing**: Explore techniques like bag-of-words, TF-IDF, and word embeddings for processing and understanding text data.
