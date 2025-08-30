# Active Learning with PyTorch on CIFAR-10 Images

This project demonstrates active learning for image classification using PyTorch. It reduces labeling effort by querying uncertain samples from CIFAR-10.

## Setup
1. Clone the repo: `git clone https://github.com/yourusername/active-learning-pytorch-images.git`
2. Create venv: `python -m venv venv`
3. Activate: `source venv/bin/activate` (or Windows equivalent)
4. Install deps: `pip install -r requirements.txt`
5. Run: `python src/active_learning_cifar.py`

## Results
- Starts with 10% labels, adds 5% per iteration.
- Tracks test accuracy.

## Extensions
- Add plots with Matplotlib.
- Try other datasets like Stanford Dogs.

For more, see the code comments.