# Assignment 2 - Tiny Neural Network

This project is a simple neural network classifier written in pure Python. It trains on a small CSV dataset with two input features, predicts a binary label, and includes a command-line menu for training, testing, prediction, saving, and loading model weights.

## Project Files

| File | Description |
| --- | --- |
| `Main.py` | Starts the command-line menu and connects all modules. |
| `network.py` | Defines the `TinyNN` model, activation functions, forward pass, backpropagation, and weight save/load methods. |
| `Train.py` | Contains training and testing functions. |
| `utils.py` | Loads the CSV dataset. |
| `data.csv` | Sample training data with `x`, `y`, and `label` columns. |

## Requirements

- Python 3.x
- No external Python packages are required.

## How to Run

From the project folder, run:

```bash
python Main.py
```

The program shows this menu:

```text
1. Train
2. Test
3. Predict
4. Save
5. Load
6. Quit
```

## Menu Options

- `Train`: Trains the neural network on `data.csv`.
- `Test`: Tests the current model on the same dataset and prints accuracy.
- `Predict`: Accepts two numeric inputs and prints the model prediction.
- `Save`: Saves the current model weights to `weights.txt`.
- `Load`: Loads previously saved weights from `weights.txt`.
- `Quit`: Exits the program.

## Dataset Format

The dataset must be a CSV file named `data.csv` with this format:

```csv
x,y,label
0.1,0.2,0
0.8,0.9,1
```

- `x` and `y` are numeric input features.
- `label` is the expected class, either `0` or `1`.

## Model Details

The neural network has:

- 2 input values
- 1 hidden layer with 3 neurons
- ReLU activation in the hidden layer
- Sigmoid activation for the output
- Binary cross-entropy loss during training

## Notes

- The model starts with random weights, so results can vary between runs.
- The `weights.txt` file is created only after choosing the `Save` option.
- This project is intended as a small educational example of neural network training without using machine learning libraries.
