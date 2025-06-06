Implementing a neural network from scratch to apply [Lesson 3](https://course.fast.ai/Lessons/lesson3.html) or [Chapter 4](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) of the fast.ai course "Practical Deep Learning for Coders".

Going to create a tutorial for myself (and perhaps for others) to learn how to build a neural network from scratch. Inspiration from this [SQLite Clone Tutorial](https://cstack.github.io/db_tutorial/)

## Key Terminology

- **Neural Network** - A computational model inspired by the human brain's structure
- **Neuron** - A single unit that applies a transformation to its inputs
- **Weights & Biases** - Parameters that are adjusted during training
- **Activation Function** - Non-linear function that introduces complexity into the model
- **ReLU** - Rectified Linear Unit, a common activation function: f(x) = max(0, x)
- **Backpropagation** - Algorithm to calculate gradients used to update weights 
- **Forward Pass** - Process of moving data through the network to get predictions
- **Gradient Descent** - Optimization method for finding weights that minimize error
- **Loss Function** - A measure of how far our predictions are from the actual data.
- **Epoch** - One complete pass through the input data

## Project Structure

### Overview

1. Linear Model & Gradient Descent
2. Non-Linearity (w/ ReLU) & Multi-Layer Networks
3. Matrix Operations & Completing the Network
4. (Optional) Testing & Extensions

### Part 1: Linear Model & Gradient Descent

- Set up project
- Created a simple linear model
- Implemented gradient descent for parameter updates
- Trained the model and begun visualizing results

### Part 2: Non-Linearity & Multi-Layer Networks

- Adding non-linearity via ReLU, which is the first step toward neural networks
- Understand activation functions, adding multiple layers is a logical extension
- Use backpropagation concepts which build directly on our gradient descent work

### Part 3: Matrix Operations & Complete Network

- Matrix operations make multi-layer networks efficient
- They enable batch processing of multiple examples
- Implement matrix ops so we can build a flexible, complete neural network class

### Part 4: Testing & Extensions

This remains as its own section for exploring and extending our neural network:

- Testing on standard datasets (e.g., MNIST)
- Adding regularization to prevent overfitting
- Implementing additional features like momentum, learning rate scheduling
- Comparing performance to frameworks like TensorFlow/PyTorch

## Project Goals

- Create a simple neural network from scratch
- Understand the fundamental components
- Build incrementally to see how each piece contributes

## TODOs

refactor writing to include:

- steps the reader should take
- steps I took (hidden behind spoiler filter)
- more "tutorial" friendly structure and language

## Some useful commands for myself

#### Convert python file to jupyter notebook
`jupytext --to notebook part2.py --output part2.ipynb` 
