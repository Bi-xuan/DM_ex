# Day-2 experiment log

For every experiment, write your prediction before running it.

## Environment
- Python / PyTorch versions:
- Device and GPU model:
- Git revision / seed / settings:

## Baseline
- What do beta, alpha and alpha_bar represent?
- Why does training sample a timestep instead of simulating all earlier steps?
- Which tensors are inputs, and which tensor is the training target?
- Why need not noise-prediction loss reach zero?
- Generated cluster proportions and within-cluster spread:

## Experiment 1: remove timestep input
- Prediction:
- Change and settings:
- Observation:
- Explanation:

## Experiment 2: reduce T to 100 with unchanged beta endpoints
- Prediction (calculate terminal alpha_bar):
- Change and settings:
- Observation:
- Explanation:

## Experiment 3: remove randomness from every reverse step
- Prediction:
- Change and settings:
- Observation:
- Explanation (why is this not automatically DDIM?):

## Understanding check
- Why start generation from fresh Gaussian noise?
- Why is subtracting predicted noise directly insufficient?
- What evidence shows that all four modes were learned?
- What differs between denoising known points and generating new points?
