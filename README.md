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

1. Linear Model & Gradient Descent
2. Non-Linearity (w/ ReLU) & Multi-Layer Networks
3. Matrix Operations & Completing the Network
4. (Optional) Testing & Extensions

## Project Goals

- Create a simple neural network from scratch
- Understand the fundamental components
- Build incrementally to see how each piece contributes

## TODOs

[] refactor writing to include:

- steps the reader should take
- steps I took (hidden behind spoiler filter)
- more "tutorial" friendly structure and language

## Some useful commands for myself

#### Convert python file to jupyter notebook
`jupytext --to notebook part2.py --output part2.ipynb` 
