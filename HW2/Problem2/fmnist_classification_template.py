import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load files using numpy
Xtrain=np.load("/Users/pegah/Projects/DS:CS541/HW2/Problem2/fashion_mnist_train_images.npy")
Ytrain=np.load("/Users/pegah/Projects/DS:CS541/HW2/Problem2/fashion_mnist_train_labels.npy")
Xtest=np.load("/Users/pegah/Projects/DS:CS541/HW2/Problem2/fashion_mnist_test_images.npy")
Ytest=np.load("/Users/pegah/Projects/DS:CS541/HW2/Problem2/fashion_mnist_test_labels.npy")
#Xtrain=Xtrain/255.0 - 0.5
#Xtest=Xtest/255.0 - 0.5

def to_one_hot(labels):
    """Convert integer labels to one-hot encoding; returns one-hot array."""
    num_classes = len(set(labels))
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels - 1] = 1
    return one_hot

#Ytrain=to_one_hot(Ytrain)
#Ytest=to_one_hot(Ytest)

def softmax(z):
    """Compute row-wise softmax; returns probability array."""
    # BEGIN YOUR CODE HERE (~1-3 lines)
    expz=np.exp(z)
    proba=expz/np.sum(expz,axis=1,keepdims=True)
    # END YOUR CODE HERE
    return proba


def cross_entropy_loss(W, b, images, labels):
    """Compute cross-entropy loss for predictions; returns scalar loss."""
    # BEGIN YOUR CODE HERE (~2-6 lines)
    z=images@W+b
    yhat=softmax(z)
    loss=-np.sum(labels*np.log(yhat))/len(labels)
    # END YOUR CODE HERE
    return loss

def compute_gradient(W, b, images, labels):
    """Compute gradients w.r.t. weights and bias; returns (dW, db)."""
    # BEGIN YOUR CODE HERE (~4-7 lines)
    z=images@W+b
    yhat=softmax(z)
    dW=images.T@(yhat-labels)/len(labels)
    db=np.sum(yhat-labels, axis=0)/len(labels)
    # END YOUR CODE HERE
    return dW, db

def compute_accuracy(W, b, images, labels):
    """Compute classification accuracy; returns fraction correct."""
    # BEGIN YOUR CODE HERE (~2-5 lines)
    z=images@W+b
    yhat=softmax(z)
    #predicted class with highest probability
    pred=np.argmax(yhat,axis=1)
    truelabel=np.argmax(labels,axis=1)
    #mean of true (1) and false (0)
    acc=np.mean(pred==truelabel)
    # END YOUR CODE HERE
    return acc
            
def show_weights(W):
    """Render weights as image patches; returns None."""
    img_size = int(W.shape[0] ** 0.5)
    canvas = np.zeros((img_size, img_size * W.shape[1]))
    for idx, col in enumerate(range(0, canvas.shape[1], img_size)):
        canvas[:, col:col+img_size] = np.reshape(W[:-1, idx], (img_size, img_size))
    plt.imshow(canvas, cmap='gray')
    plt.show()

def train_softmax_classifier(train_images, train_labels, val_images, val_labels,
                             learning_rate=1e-5, batch_size=16, nepochs=100):
    """Train softmax classifier; returns (W, b, loss, accuracy)."""
    num_batches = train_images.shape[0] // batch_size
    n_features = train_images.shape[1]
    n_classes = train_labels.shape[1]

    # Initialize weights and bias
    # BEGIN YOUR CODE HERE (~2 lines)
    W=np.zeros((n_features,n_classes))
    b=np.zeros(n_classes)
    # END YOUR CODE HERE

    cost, acc = float('inf'), 0.0
    for epoch in range(nepochs):
        np.random.seed(epoch)
        perm = np.random.permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        for batch in range(num_batches):
            # what do you need to do in each iteration?
            # BEGIN YOUR CODE HERE (~5 lines)
            start=batch*batch_size #start index
            end=start+batch_size #end index
            x_batch=train_images[start:end] #images based on a batch
            y_batch=train_labels[start:end] #labels based on a batch
            dW,db=compute_gradient(W, b, x_batch,y_batch)
            W-=learning_rate*dW
            b-=learning_rate*db

            # END YOUR CODE HERE
        # what do you need to compute at the end of each epoch?
            # we need to compute model performance on validation. 
        # BEGIN YOUR CODE HERE (~3 lines)
        cost=cross_entropy_loss(W,b,val_images,val_labels) #loss
        acc=compute_accuracy(W,b,val_images,val_labels) #accuracy
        # END YOUR CODE HERE
    return W, b, cost, acc

# def add_bias(images):
#     return np.hstack((images, np.ones((images.shape[0], 1))))

if __name__ == "__main__":
    # Load data
    training_images = np.load("fashion_mnist_train_images.npy") / 255.0 - 0.5
    training_labels = to_one_hot(np.load("fashion_mnist_train_labels.npy"))
    testing_images = np.load("fashion_mnist_test_images.npy") / 255.0 - 0.5
    testing_labels = to_one_hot(np.load("fashion_mnist_test_labels.npy"))

    # split training into training and validation using train_test_split
    x_train, x_val, y_train, y_val = train_test_split(training_images, training_labels,
                                                      test_size=0.2, random_state=541)

    # List hyperparameters to try
    # BEGIN YOUR CODE HERE (~2-3 lines)
    learning_rates=[1e-1]
    #tries with different values of: 0.1, 0.01, 0.001, 1e-05, 1e-07
    batch_sizes=[32]
    #tries with different values of: 2,32,64,128


    # END YOUR CODE HERE

    # Initialize varjables to keep track of best hyperparameters
    # BEGIN YOUR CODE HERE (~3 lines)
    best_acc=0.0
    best_lr=None
    best_bs=None
    # END YOUR CODE HERE

    # Train model
    # BEGIN YOUR CODE HERE (~7-10 lines)
    for lr in learning_rates: #when we give a list of learning rates
            for bs in batch_sizes:  #when we give a list of batch sizes
                W, b, val_cost, val_acc=train_softmax_classifier(
                    x_train,y_train,x_val,y_val, #train and val data
                    learning_rate=lr,batch_size=bs,nepochs=20)
                
                if val_acc>best_acc: #getting best(if doing a list of hyperparam)
                    best_acc=val_acc
                    best_lr=lr
                    best_bs=bs
    # END YOUR CODE HERE

    # Retrain model on full training set with best hyperparameters and evaluate on test set
    # BEGIN YOUR CODE HERE (~1 line)
    #final_W, final_b, loss, acc = train_softmax_classifier(training_images, training_labels, testing_images, testing_labels,
                                   #learning_rate=best_lr, batch_size=best_bs, nepochs=10, wandb_init=False)
    final_W, final_b, loss, acc = train_softmax_classifier(training_images, training_labels, testing_images, testing_labels,
                                   learning_rate=best_lr, batch_size=best_bs, nepochs=10)
    # END YOUR CODE HERE
print(f"Learning Rate:{best_lr}")
print(f"Batch Size: {best_bs}")
print(f"val Accuracy:{best_acc * 100:.2f}%")
print(f"test Accuracy:{acc * 100:.2f}%")
print(f"test loss:{loss:.4f}")
    # showWeights(lr[:,:])