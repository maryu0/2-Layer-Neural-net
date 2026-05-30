import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

def one_hot_encoded(y, num_classes=10):
    m=y.shape[0]
    oh=np.zeros((m,num_classes))
    oh[np.arange(m),y]=1
    return oh

Y_train=one_hot_encoded(y_train)
Y_test=one_hot_encoded(y_test)

def init_params(layer_dims):
    np.random.seed(42)
    params={}

    params['W1']=np.random.randn(layer_dims[0],layer_dims[1])*0.01 # 0.01 to avoid exploding gradients
    params['b1']=np.zeros((1,layer_dims[1]))

    params['W2']=np.random.randn(layer_dims[1],layer_dims[2])*0.01
    params['b2']=np.zeros((1,layer_dims[2]))
    return params

def relu(Z):
    return np.maximum(0,Z)

def relu_derivative(Z):
    return (Z>0).astype(float)

def softmax(Z): 
    # For numerical stability we shift Z by its max val
    Z_shifted=Z-np.max(Z,axis=1,keepdims=True) # keepdims=True to maintain shape for broadcasting
    exp_Z=np.exp(Z_shifted)
    return exp_Z/np.sum(exp_Z,axis=1,keepdims=True)

def forward_pass(X, params):
    W1, b1=params['W1'],params['b1']
    W2, b2=params['W2'],params['b2']

    Z1=X@W1+b1      # shape (m,128)
    A1=relu(Z1)     # shape(m,128)

    Z2=A1@W2+b2     # shape(m,10)
    A2=softmax(Z2)  # shape(m,10)
 
    cache={'X':X,'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2}
    return A2, cache

def cross_entropy_loss(A2, Y):
    m=Y.shape[0]
    A2_clipped=np.clip(A2, 1e-15, 1-1e-15) # to avoid log(0)
    loss=-np.sum(Y*np.log(A2_clipped))/m
    return loss

# Back-propagation
def backward_pass(cache, params, Y):
    m=Y.shape[0]
    W2=params['W2']
    Z1=cache['Z1']
    A1=cache['A1']
    A2=cache['A2']
    X=cache['X']

    dZ2=A2-Y
    dW2=A1.T@dZ2/m
    db2=np.sum(dZ2, axis=0, keepdims=True)/m

    dA1=dZ2@W2.T
    dZ1=dA1*relu_derivative(Z1)
    dW1=X.T@dZ1/m
    db1=np.sum(dZ1, axis=0, keepdims=True)/m

    grads = {'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2}
    return grads

def update_params(params, grads, lr):
    params['W1']-=lr*grads['dW1']
    params['b1']-=lr*grads['db1']
    params['W2']-=lr*grads['dW2']
    params['b2']-=lr*grads['db2']

    return params

# Training loop
def train(X, Y, layer_dims, lr=0.1, epochs=50, batch_size=256):
    params=init_params(layer_dims)
    m=X.shape[0]
    loss_history=[]

    for epoch in range(epochs):
        # 1. Shuffle
        indices=np.random.permutation(m)
        X_shuffled=X[indices]
        Y_shuffled=Y[indices]

        epoch_loss=0
        num_batches=0

        for i in range(0,m,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            Y_batch=Y_shuffled[i:i+batch_size]

            # Forward pass
            A2, cache=forward_pass(X_batch, params)

            # loss
            loss=cross_entropy_loss(A2, Y_batch)

            # Backward pass
            grads=backward_pass(cache,params,Y_batch)

            # update params
            params=update_params(params,grads,lr)

            epoch_loss+=loss
            num_batches+=1
        
        avg_loss=epoch_loss/num_batches
        loss_history.append(avg_loss)

        # Computing acc for few epochs
        if epoch%5==0:
            preds=np.argmax(forward_pass(X,params)[0], axis=1)
            labels=np.argmax(Y, axis=1)
            acc=np.mean(preds==labels)
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

    return params, loss_history

params, history = train(X_train, Y_train, [784, 128, 10], lr=0.1, epochs=50)

def evaluate(X, Y, params):
    A2, _ = forward_pass(X, params)
    preds = np.argmax(A2, axis=1)
    labels = np.argmax(Y, axis=1)
    return np.mean(preds == labels)

test_acc = evaluate(X_test, Y_test, params)
print(f"Test accuracy: {test_acc:.4f}")

# Plot loss curve
plt.plot(history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.savefig('loss_curve.png')
plt.show()