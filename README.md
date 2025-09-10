# Active Learning with PyTorch on CIFAR-10 Images

This project demonstrates active learning for image classification using PyTorch. It reduces labeling effort by querying uncertain samples from CIFAR-10.

It's based of the following 5 strategies:

1) Baseline: 'Random'
   
What it does: Randomly selects samples without any strategy
When to use: Baseline comparison; when data is uniformly distributed
Pros: Simple, unbiased, no computational overhead
Cons: Inefficient use of labeling budget

2 - 5 are meant to improve performance over baseline:

2) Least Confidence

What it does: Selects samples where the model's highest class probability is lowest
Example: If model predicts [0.4, 0.3, 0.3], uncertainty = 1 - 0.4 = 0.6
When to use: Multi-class problems where you want simple uncertainty
Pros: Fast, intuitive, works well for balanced datasets
Cons: Only considers top prediction, ignores other class information
Python Algorithm: uncertainty = 1 - max(P(y|x))

3) Entropy Sampling,

What it does: Measures overall uncertainty across all classes
Example: Uniform [0.33, 0.33, 0.34] has high entropy; [0.95, 0.03, 0.02] has low entropy
When to use: When you want to consider full probability distribution
Pros: Uses all class information, theoretically grounded
Cons: Can select outliers, treats all uncertainty equally

Python Algorithm: H(x) = -Σ P(y|x) × log(P(y|x))
   
7) BALD (Bayesian Active Learning by Disagreement),

What it does: Measures model uncertainty vs data uncertainty using MC-Dropout
How it works:
Runs multiple forward passes with dropout
Measures disagreement between predictions
High disagreement = high epistemic uncertainty = worth learning
When to use: When you want to distinguish "model doesn't know" from "data is ambiguous"
Pros: Theoretically principled, focuses on reducible uncertainty
Cons: Computationally expensive, can overselect outliers (as seen in your results)
Algorithm:
BALD = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]
     = Total Uncertainty - Aleatoric Uncertainty
     = Epistemic Uncertainty

8) Diverse + Entropy (Diversity-Aware Sampling)

What it does: Balances uncertainty with diversity to avoid redundant selections
Why it matters: Pure uncertainty might select 100 similar dog images; diversity ensures variety
When to use: When uncertain samples might be correlated/similar
Pros: Avoids redundancy, better coverage of data space
Cons: More complex, requires good feature representations

Divesity Algorithm:
1. Select 3×n_samples most uncertain samples
2. Cluster them in feature space
3. Pick most uncertain from each cluster

## Project Setup:

1. Clone the repo: `git clone https://github.com/yourusername/active-learning-pytorch-images.git`
2. Create venv: `python -m venv venv`
3. Activate: `source venv/bin/activate` (or Windows equivalent)
4. Install deps: `pip install -r requirements.txt`
5. Run: `python src/active_learning_cifar.py`

## Results
- Starts with 10% labels, adds 5% per iteration.
- Tracks test accuracy.
- This project shows a decrease in the amount of images in training data to hit the same accuracy of 10 - 20 percent.

<img width="560" height="353" alt="image" src="https://github.com/user-attachments/assets/5a7e0455-115f-4cf6-8914-e4999d8371fd" />
<img width="156" height="105" alt="image" src="https://github.com/user-attachments/assets/0d5160f3-3d35-4b31-a889-057e643ccb2c" />

## Extensions
- Add plots with Plotly.

For more, see the code comments.
