# Neural Digits

A simple Python script for training, testing, and performing inference with a neural network on the MNIST dataset of handwritten digits. Built with PyTorch, Neural Digits allows for training from scratch, continuing from a saved model, evaluating performance, and predicting digits from individual image files.

## Features

- **Training Mode:** Train a neural network from scratch or continue training from a saved checkpoint. Capable of achieving over 99% accuracy on the MNIST training data with about 50 epochs of training.
- **Testing Mode:** Evaluate the trained model's accuracy on the MNIST test dataset.
- **Inference Mode:** Predict the digit in a provided image file.

## Prerequisites

Before running this script, you will need the following:
- Python 3.6 or higher
- PyTorch
- torchvision
- PIL
- numpy
- tqdm

These packages can be installed using `pip`:

```bash
pip install torch torchvision numpy Pillow tqdm
```

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/Zelaron/neural-digits.git
cd neural-digits
```

## Usage

Run the script from the command line by navigating to the repository's directory and executing the main script:

```bash
python neural_digits.py
```

You will be prompted to enter a program mode:
- Enter `0` for training the model.
- Enter `1` to test the model's performance on the MNIST test dataset.
- Enter `2` to perform inference on a single image file named `digit.png`.

Follow the on-screen prompts to further configure each mode.

## Contributing

Contributions to this project are welcome! Here are a few ways you can help:
- Report bugs and issues.
- Suggest improvements or new features.
- Improve the existing code to enhance performance or readability.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.
