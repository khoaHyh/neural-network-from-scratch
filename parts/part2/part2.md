# Neural Network from Scratch - Part 2: Non-Linearity & Multi-Layer Networks

For Part 2, I'll be adding non-linearity through activation functions and expanding to multiple layers—the key components that make a neural network powerful. Whatever that means 🤷🏻‍♂️.

## Prep

### Learning Objectives for Part 2

- Understand why non-linearity is essential in neural networks
- Implement the ReLU activation function and its derivative
- Build a multi-layer architecture with hidden layers
- Extend backpropagation to work across multiple layers
- Train our network on non-linearly separable data

### Research Areas

To help with my own research, I'll explore these topics:

- **Why non-linearity matters**: What happens if you stack linear layers without activation functions?
- **ReLU vs. other activation functions**: How does ReLU compare to sigmoid or tanh?
- **Vanishing/exploding gradients**: Why can these be problems in deep networks?
- **Backpropagation algorithm**: Understand the chain rule as it applies to neural networks
- **Hidden layer neurons**: How does the number of neurons affect network capacity?

### Excel Mapping (Connecting to Jeremy Howard's Approach)

To relate back to the Excel-based model:

- ReLU would be an IF statement in Excel: `=IF(A1>0,A1,0)`
- Multiple layers would be multiple sheets or sections, each with its own weights
- Backpropagation would calculate how each cell's value affects the final error

## Findings
