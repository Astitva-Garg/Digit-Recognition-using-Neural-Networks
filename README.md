# Handwritten Digit Classification – From Scratch and with PyTorch  

This project implements a **handwritten digit classifier** (MNIST-like dataset) in two ways:  

1. **Custom Numpy Implementation** – A neural network coded from scratch with full control over feedforward, backpropagation, parameter updates, and training loops.  
2. **PyTorch Implementation** – A cleaner, modular version using PyTorch’s high-level abstractions (datasets, dataloaders, optimizers, loss functions, etc.).  

The aim is to **compare low-level neural network implementation** with a **high-level deep learning framework**, helping learners understand the mathematics and mechanics behind deep learning while also practicing production-ready PyTorch coding.  

---

## Features
- **From Scratch (NumPy version):**
  - Implements ReLU and Softmax activation functions  
  - Manual backpropagation for weight updates  
  - One-hot encoding for labels  
  - Gradient descent optimization  
  - Training loop with live accuracy tracking  

- **PyTorch Version:**
  - Modular `nn.Module` class for defining layers  
  - Data handling with `TensorDataset` + `DataLoader`  
  - GPU acceleration (CUDA if available)  
  - CrossEntropy loss & Adam optimizer  
  - Evaluation loop with accuracy reporting  

---

## Project Structure
```
├── Data/
│   └── train.csv     # Training + testing data (similar format to MNIST Kaggle dataset)
├── numpy_nn.py        # NumPy-only neural network
├── torch_nn.py        # PyTorch neural network
├── README.md          # Project description
```

---

## 📊 Dataset
- The project uses the **MNIST digit dataset format** (as found on Kaggle's Digit Recognizer competition).  
- `train.csv` contains pixel values for 28x28 grayscale images (flattened into 784 features).  
- The first column is the digit label (0–9), followed by pixel intensities (0–255).  

---

## Running the Models

### 1. NumPy Implementation
```bash
python numpy_nn.py
```
- Trains a 2-layer neural network for classification.  
- Prints iteration progress and training accuracy.  

### 2. PyTorch Implementation
```bash
python torch_nn.py
```
- Uses `DataLoader` for mini-batches.  
- Runs on **GPU if available**.  
- Prints training loss during epochs and final test accuracy.  

---

## Results
- NumPy network achieves decent accuracy (but slower, manually optimized).  
- PyTorch model converges faster, is more scalable, and achieves higher accuracy due to better optimization and batching.  

---

## Learning Goals
- Understand how **feedforward** and **backpropagation** work at the matrix level.  
- Learn to implement a basic neural net with **NumPy only**.  
- See how **PyTorch simplifies training** with its abstractions.  
- Compare performance differences between custom and framework implementations.  

---

## Future Improvements
- Add more layers (deep network) in both versions  
- Implement regularization (dropout, L2)  
- Experiment with different optimizers (SGD, RMSprop, etc.)  
- Add visualization of loss & accuracy curves  

---

## License
This project is open-source under the MIT License.  
