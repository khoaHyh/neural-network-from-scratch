# Neural Network from Scratch - Part 3: Matrix Operations & Complete Network

In part 3, I'll be improving the model by adding matrix multiplication and overall making the neural network more mature.

## Prep

### Learning Objectives

- Understand how matrix operations enable batch processing and computational efficiency
- Implement vectorized forward and backward passes
- Build a flexible neural network class that can handle arbitrary layer sizes
- Understand the relationship between individual neuron computations and matrix operations

### Skills Gained

- Matrix multiplication for neural network operations
- Proper weight initialization and parameter management
- Batch gradient descent implementation
- Understanding of computational graphs and data flow

## Findings

### Batch multiplication

**Matrix multiplication is fundamentally about transforming data**. We can think of it like a recipe converter. We have some ingredients (input features) and we want to make a dish (output). The matrix contains "recipe ratios" - how much of each ingredient contributes to the final result.

- good resource: [Khan Academy vid on matrix multiplication](https://www.youtube.com/watch?v=kT4Mp9EdVqs)

Matrix multiplication is one of the most important mathematical operations in deep learning. Here are some reasons why it is significant:

- **mathematical efficiency** - instead of manually calculating the product of every neuron, it does all neurons' calculations simultaneously
- **pattern recognition** - matrices can rotate, scale and project data into new dimensional spaces which makes patterns more obvious to spot
- **hardware optimization** - our CPUs/GPUS are specifically designed for matrix operations, making them orders of magnitude faster than equivalent loop-based ops

The **batch** part comes in when we handle multiple examples simultaneously. Matrix multiplication can handle multiple neurons simultanously so we get the mathematical efficiency + hardware utilization benefits together. This is like having a factory assembly line that processes multiple products at a time instead of one-by-one.

### Vectorization

In this part, we're may be noticing a crucial insight which is that neural networks are at their core **computationally intensive**. To take our neural network to the next level, we need to tighten some things up and make them more **efficient**. Vectorization is another way to make things practically faster.

**Vectorization** is essentially replacing loops with array operations. Take these code examples:

```python
# scalar approach (we did in part 2)
for i in range(neurons):
    output[i] = 0
    for j in range(inputs):
        output[i] += input[j] * weight[i][j]
    output[i] += bias[i]
```


```python
# vectorized approach (part 3)
output = np.dot(input, weights) + bias
```

Modern neural networks can have millions/billions of parameters so our approaches to training can mean taking days (scalar approach) to minutes (vectorized approach). However, using vectorization may not be intuitive so some key steps to ensure we do it correctly are:

1. **Data organization**: structure our data so operations can work on entire array
1. **Broadcasting**: let numpy handle dimension matching automatically
1. **Memory layout**: arrange data so hardware can access it efficiently

With these things in mind, we can unlock the full potential of our computers to ensure it can handle our neural networks.

### Flexible and scalable layers

Coming from part 2 we currently have a hardcoded network. Our weights are manually defined (`weight`, `bias`, `weight2`, `bias2`). This isn't practical, slows us down and even blocks us in a couple of ways:

- **real-world flexibility** - different problems need different architectures. One use case may require 1000+ neurons per layer, while a simpler problem may require 10.
- **experimentation** - to allow us to tinker with curiousities like "does this network better with 100 neurons vs 10?", we should be able to easily test without rewriting code.
- **scalability** - to move from our simpler, toy problems to real-world datasets and use cases, we need to be able to **pragmatically** handle much more without manually defining parameters
- **code reusability** - from an engineering practice perspective, we should be able to handle any size instead of having separate implementations (having different networks with different layer sizes)

This takes our neural network from a hobby project to a usable tool. This enables us to have a tool that is applicable to real world datasets and problems.

### It's not magic 🪄

An important mental chunk to consume is that the math we came across in Parts 1-2 scales up to real neural networks. Matrix operations **aren't magic** - they are an efficient way to do the same neuron computations we did in the earlier parts but **simultaneously**.

#### Individual ➡️ Matrix

**Individual neuron**:
```python
neuron_output = ReLU(input * weight + bias)
```

**Matrix operation**:
```python
# @ is Python's matrix multiplication operator. This says multiply the inputs matrix by the weights matrix.
layer_output = ReLU(inputs @ weights + biases)
```

#### Insight

Introducing matrices, vectorization, and flexible layer sizes do not change our neural network on a fundamental level. It is still the same neural network we built, it just computes way more all at once.
