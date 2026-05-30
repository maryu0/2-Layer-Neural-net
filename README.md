# 2-Layer MNIST Neural Network (NumPy)

A simple from-scratch 2-layer neural network implemented in NumPy for classifying the MNIST dataset. The repository includes a single script that trains a small fully-connected network (784 → 128 → 10), evaluates test accuracy, and saves a training loss curve.

## Files

- [2-layer-neural-net.py](2-layer-neural-net.py): Main training and evaluation script.

## Features

- Lightweight implementation using NumPy and Keras dataset loader (no heavy framework required to understand learning internals).
- Implements forward and backward propagation, ReLU activations, softmax output and cross-entropy loss.
- Mini-batch SGD training, basic training loss logging and test evaluation.

## Requirements

This project was developed using Python 3.10+ and relies on the following packages:

- numpy
- matplotlib
- tensorflow (for the Keras MNIST dataset loader)

You can install these into a virtual environment. A sample `requirements.txt` isn't included, but the following command installs the main packages:

```bash
pip install numpy matplotlib tensorflow
```

## Setup (Windows PowerShell)

1. Create a virtual environment (optional):

```powershell
python -m venv mnist-env
```

2. Activate it:

```powershell
.\mnist-env\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install numpy matplotlib tensorflow
```

## Run training

To train the model and generate the loss curve image, run:

```powershell
python 2-layer-neural-net.py
```

The script will print training progress (every 5 epochs) and save `loss_curve.png` in the working directory.

## Evaluate

The script evaluates test accuracy after training and prints the test accuracy to stdout. You can also call the `evaluate` function from the script if importing it into another module.

## Notes and tips

- Training is intentionally minimal and educational — it uses a small network and simple hyperparameters. Change `layer_dims`, `lr`, `epochs`, and `batch_size` inside the script to experiment.
- The script downloads the MNIST dataset automatically via `tensorflow.keras.datasets.mnist`, so no manual dataset download is required.
- For faster training or experiments, consider using TensorFlow / Keras models or GPU-accelerated environments.

## License

This project is provided under the MIT License. See LICENSE for details (not included).

## Contributing

Feel free to open issues or submit pull requests to improve documentation, add tests, or modularize the code.

---

If you'd like, I can add a `requirements.txt`, a small `LICENSE` file, or commit these changes for you. Which would you prefer?
