# problem 3 solution

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x_train, y_train, w0, alpha, num_steps):
  """
  Performs gradient descent to learn the weights of a linear regression model.

  Args:
    x_train (np.ndarray): The training input features.
    y_train (np.ndarray): The training target values.
    w0 (np.ndarray): The initial weights.
    alpha (float): The learning rate.
    num_steps (int): The number of gradient descent steps.

  Returns:
    np.ndarray: The learned weights.
  """
  # PRO TIP: CREATE DEEP COPY of arrays passed as args to avoid changing them
  # Your code here (~2 lines)
  x_train_copy = np.copy(x_train)
  y_train_copy = np.copy(y_train)
  w0 = np.copy(w0)

  losses = []

  for i in range(num_steps):

    # compute loss (~2 lines)
    # Your code here
    y_hat = np.dot(x_train_copy, w0)
    loss = np.mean((y_hat - y_train_copy) ** 2)
    losses.append(loss)
    
    # Placeholder for loss, remove after implementing
    # loss = 0 
    print(f'Training loss = {loss:.3f}')

    # compute gradients and update params (~2 lines)
    # Your code here
    gradient = (2 / x_train_copy.shape[0]) * np.dot(x_train_copy.T, (y_hat - y_train_copy))
    w0 -= alpha * gradient
    w = w0


  return w, losses


def train_linear_regression(x_train, y_train, alpha=0.01):
    """
    Initializes and trains a linear regression model.

    Args:
      x_train (np.ndarray): The training input features.
      y_train (np.ndarray): The training target values.

    Returns:
      np.ndarray: The trained model weights.
    """
    # PRO TIP: setting a random seed makes the experiments reproducible (hopefully in other systems too)
    np.random.seed(541)
    
    # Initialize weights (~2 lines)
    # Your code here 

    w0 = np.random.randn(x_train.shape[1])
    # b = np.random.randn(1)

    # Placeholder for w0, remove after implementing
    # w0 = np.zeros(x_train.shape[1])
    print(f'Model coefficients: {np.round(w0,3)}')

    # Make initial prediction (~1 line)
    # Your code here
    y_pred = np.dot(x_train, w0)

    # Placeholder for y_pred, remove after implementing
    # y_pred = np.zeros(y_train.shape)
    print(y_pred[:5])

    # Calculate initial loss (~1-2 lines)
    # PRO TIP: tracking the train loss at the beginning and during optimization is
    # key for debugging, determining convergence
    # Your code here
    loss = np.mean((y_pred - y_train) ** 2)

    # Placeholder for loss, remove after implementing
    # loss = 0
    print(f'Training loss = {loss:.3f}')

    # Run gradient descent
    w, losses = gradient_descent(x_train, y_train, w0, alpha, 100)

    return w, losses

def plot_loss_curves(x_train: np.ndarray, y_train: np.ndarray, alphas: list[float]):
    """
    Plots the loss curves for different learning rates.

    Args:
      alphas (list of float): The learning rates to evaluate.
    """
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        # Train model with the given alpha
        w, losses = train_linear_regression(x_train, y_train, alpha=alpha)
        plt.plot(losses, label=f'alpha={alpha}')
    
    plt.title('Loss Curves for Different Learning Rates')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('HW1/loss_curves.pdf')

def main():
    """
    Main function to load data, pre-process, train, and evaluate the model.
    """
    # Load data from files
    # Make sure you have the .npy files in the same directory as this script
    try:
        X_tr = np.reshape(np.load("HW1/age_regression/age_regression_Xtr.npy"), (-1, 48*48))
        ytr = np.load("HW1/age_regression/age_regression_ytr.npy")
        X_te = np.reshape(np.load("HW1/age_regression/age_regression_Xte.npy"), (-1, 48*48))
        yte = np.load("HW1/age_regression/age_regression_yte.npy")
        print(X_tr.shape, ytr.shape, X_te.shape, yte.shape)
    except FileNotFoundError:
        print("Dataset files not found. Please download them and place them in the same directory.")
        return

    # (optional) Pre-processing

    # PRO TIP: center and normalize the features from training set (~4 lines)
    # Your code here

    print("Normalizing the features...")
    # train_mean = np.mean(X_tr, axis=0)
    # train_std = np.std(X_tr, axis=0)
    # train_std[train_std == 0] = 1.0

    # X_tr = (X_tr - train_mean) / train_std
    # X_te = (X_te - train_mean) / train_std
    print("Feature normalization complete.")
    

    # PRO TIP: prepend a 1 to each of the observations in X_tr and X_te (~2 lines)
    # This way you can treat the bias just as a regular weight
    # Your code here
    print("Prepending a 1 to each observation in X_tr and X_te...")
    X_tr = np.hstack((np.ones((X_tr.shape[0], 1)), X_tr))
    X_te = np.hstack((np.ones((X_te.shape[0], 1)), X_te))
    print("Prepending complete.")
    print(f'X_tr shape after prepending: {X_tr.shape}, X_te shape after prepending: {X_te.shape}')

    # Train model
    print("Training the linear regression model...")
    w, losses = train_linear_regression(X_tr, ytr, alpha=2e-4)

    # Report fMSE cost on the training and testing data (separately)
    # Your code here (~4-6 lines)
    train_loss = np.mean((np.dot(X_tr, w) - ytr) ** 2)
    test_loss = np.mean((np.dot(X_te, w) - yte) ** 2)
    print(f'Training loss after training = {train_loss:.3f}')
    print(f'Testing loss after training = {test_loss:.3f}')
    plot_loss_curves(X_tr, ytr, alphas=[1e-4, 2e-4, 5e-5])
    


# Run main
if __name__ == "__main__":
    # X_tr = np.reshape(np.load("HW1/age_regression/age_regression_Xte.npy"), (-1, 48*48))
    # print(X_tr.shape)
    main()

