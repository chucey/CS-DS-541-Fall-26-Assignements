import numpy as np

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
  pass

  for i in range(num_steps):

    # compute loss (~2 lines)
    # Your code here
    pass
    
    # Placeholder for loss, remove after implementing
    loss = 0 
    print(f'Training loss = {loss:.3f}')

    # compute gradients and update params (~2 lines)
    # Your code here
    pass

  return w


def train_linear_regression(x_train, y_train):
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
    pass

    # Placeholder for w0, remove after implementing
    w0 = np.zeros(x_train.shape[1])
    print(f'Model coefficients: {np.round(w0,3)}')

    # Make initial prediction (~1 line)
    # Your code here
    pass

    # Placeholder for y_pred, remove after implementing
    y_pred = np.zeros(y_train.shape)
    print(y_pred[:5])

    # Calculate initial loss (~1-2 lines)
    # PRO TIP: tracking the train loss at the beginning and during optimization is
    # key for debugging, determining convergence
    # Your code here
    pass

    # Placeholder for loss, remove after implementing
    loss = 0
    print(f'Training loss = {loss:.3f}')

    # Run gradient descent
    w = gradient_descent(x_train, y_train, w0, 2e-3, 100)

    return w

def main():
    """
    Main function to load data, pre-process, train, and evaluate the model.
    """
    # Load data from files
    # Make sure you have the .npy files in the same directory as this script
    try:
        X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
        ytr = np.load("age_regression_ytr.npy")
        X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
        yte = np.load("age_regression_yte.npy")
        print(X_tr.shape, ytr.shape, X_te.shape, yte.shape)
    except FileNotFoundError:
        print("Dataset files not found. Please download them and place them in the same directory.")
        return

    # (optional) Pre-processing

    # PRO TIP: center and normalize the features from training set (~4 lines)
    # Your code here
    pass
    

    # PRO TIP: prepend a 1 to each of the observations in X_tr and X_te (~2 lines)
    # This way you can treat the bias just as a regular weight
    # Your code here
    pass

    # Train model
    w = train_linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # Your code here (~4-6 lines)
    pass


# Run main
if __name__ == "__main__":
    main()
