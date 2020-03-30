import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
def gradient_descent(g, alpha, max_its, w):
    # compute the gradient of our input function - note this is a function too!
    weight_history=[]
    gradient = grad(g)
    W=[]

    # run the gradient descent loop
    best_w = w  # weight we return, should be the one providing lowest evaluation
    best_eval = g(w)  # lowest evaluation yet
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha * grad_eval

        # return only the weight providing the lowest evaluation
        test_eval = g(w)
        weight_history.append(test_eval)
        W.append(k)

        if test_eval < best_eval:
            best_eval = test_eval
            best_w = w

    return W,weight_history
g = lambda w: 1/float(50)*(w**4 + w**2 + 10*w)
W,weight_history = gradient_descent(g = g,alpha = 10**-2,max_its = 2000,w = 2.0)


def cost_history(W,weight_history):
    # loop over weight history and compute associated cost function history at each step
    plt.figure(num=1,figsize=(8,5))
    plt.plot(W,weight_history)
    plt.show()

cost_history(W,weight_history)
#for i in range (4):
