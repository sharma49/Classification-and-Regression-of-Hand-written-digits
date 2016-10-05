import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import pickle

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """
    # print("Loading Dataset")

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # print("Train Data ",train_data.shape, " Train Label ", train_label.shape)
    # print("Validation Data ",validation_data.shape, " Validation Label ", validation_label.shape)
    # print("Test Data ",test_data.shape, " Test Label ", test_label.shape)

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    # print("Train Data ",train_data.shape, " Train Label ", train_label.shape)
    # print("Validation Data ",validation_data.shape, " Validation Label ", validation_label.shape)
    # print("Test Data ",test_data.shape, " Test Label ", test_label.shape)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    # error = 0
    # error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # print("BLR Training Data Shape : ", train_data.shape)
    # print("Labeli Value ", labeli)
    w = initialWeights.reshape((n_feature+1,1))
    # print(w[0:10, :])
    # print("IW Shape ", initialWeights.shape)
    bias_term = np.ones((n_data,1))
    x = np.hstack((bias_term,train_data))
    # print("X Shape ", x.shape)
    # print("Dot Value Shape ",dot_val.shape)
    # print("Dot Value ", dot_val[0:5,:])
    theta = sigmoid(np.dot(x,w))
    # print("Theta Shape ", theta.shape)
    # print("Labeli Shape ", labeli.shape)
    # print("Theta Value ", theta[0:5,:])
    error = labeli * np.log(theta) + (1.0 - labeli) * np.log(1.0 - theta)
    error = -np.sum(error) / n_data
    # print("Error value ",error)
    error_grad = (theta - labeli) * x
    error_grad = np.sum(error_grad, axis=0) / n_data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    # label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias_term = np.ones((data.shape[0], 1))
    x = np.hstack((bias_term,data))

    label = sigmoid(np.dot(x, W))
    label = np.argmax(label, axis=1)
    label = label.reshape((data.shape[0],1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        Y: the label vector of size N x k where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, Yi = args
    # print("Shape of Yi ",Yi.shape)
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    # error = 0
    # error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    # print("IW Shape ", params.shape)
    w = params.reshape((n_feature+1, n_class))
    # print("Weight Shape ", w.shape)
    bias_term = np.ones((n_data,1))
    x = np.hstack((bias_term,train_data))
    # print("X Shape", x.shape)
    exponent_val = np.exp(np.dot(x, w))
    # print("EXP Shape", exponent_val.shape)
    # print("Summation Shape ",np.tile(np.sum(exponent_val, axis = 1), (10,1)).T.shape)
    post_prob = exponent_val / np.tile(np.sum(exponent_val, axis = 1), (10,1)).T
    # print("Post Prob Shape ", post_prob.shape)
    error = Yi * np.log(post_prob)
    # print("Error 1 Shape", error.shape)
    error = -np.sum(error) / n_data
    # print("Error ", error)

    error_grad = np.dot(np.transpose(x), post_prob - Yi) / n_data
    error_grad = error_grad.flatten()
    # error_grad = np.sum(error_grad) / n_data
    # print("EG Shape ",error_grad.shape)
    # error_grad = error_grad.flatten()
    # print("Error Grad Shape ", error_grad.shape)
    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    # label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias_term = np.ones((data.shape[0], 1))
    x = np.hstack((bias_term,data))
    label = np.exp(np.dot(x, W)) / np.sum(np.exp(np.dot(x, W)))
    # print("Label Shape ", label.shape)
    label = np.argmax(label, axis=1)
    label = label.reshape((data.shape[0],1))

    # for x in range(data.shape[0]):
    #     max_index = np.argmax(label[x])
    #     label[x] = max_index
    return label


"""
Script for Logistic Regression
"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
# print("Y Before",Y[0, :])
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()
# print("Y After",Y[0, :])
# Logistic Regression with Gradient Descent

W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# with open('params.pickle', 'wb') as f1:
#     pickle.dump(W, f1)

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# with open('params_bonus.pickle', 'wb') as f2:
#     pickle.dump(W_b, f2)

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
train_label= train_label.reshape(train_label.shape[0])
test_label= test_label.reshape(test_label.shape[0])
validation_label= validation_label.reshape(validation_label.shape[0])

##################################################################

print('\n linear kernel : ')
clf = SVC( kernel='linear')
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)

##################################################################

print('\n rbf kernel and gamma = 1.0 : ')
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)

##################################################################

print('\n rbf kernel and default gamma : ')
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)

##################################################################

print('\n rbf kernel and default gamma (C=1) : ')
clf = SVC(kernel='rbf', C=1)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)

##################################################################

print('\n rbf kernel and default gamma (C=10) : ')
clf = SVC(kernel='rbf', C=10)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=20) : ')
clf = SVC(kernel='rbf', C=20)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=30) : ')
clf = SVC(kernel='rbf', C=30)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)

##################################################################

print('\n rbf kernel and default gamma (C=40) : ')
clf = SVC(kernel='rbf', C=40)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=50) : ')
clf = SVC(kernel='rbf', C=50)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=60) : ')
clf = SVC(kernel='rbf', C=60)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=70) : ')
clf = SVC(kernel='rbf', C=70)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=80) : ')
clf = SVC(kernel='rbf', C=80)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)
##################################################################

print('\n rbf kernel and default gamma (C=90) : ')
clf = SVC(kernel='rbf', C=90)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)

##################################################################

print('\n rbf kernel and default gamma (C=100) : ')
clf = SVC(kernel='rbf', C=100)
clf.fit(train_data, train_label)

output_train_data = clf.predict(np.array(train_data))
result_train_data = (output_train_data == train_label)
result_float_train_data = result_train_data.astype(np.float)
average_train_data = np.average(result_float_train_data)
accuracy_train_data = str(100*average_train_data)

print('\n Accuracy on training set : ' + accuracy_train_data)

output_validation_data = clf.predict(np.array(validation_data))
result_validation_data = (output_validation_data == validation_label)
result_float_validation_data = result_validation_data.astype(np.float)
average_validation_data = np.average(result_float_validation_data)
accuracy_validation_data = str(100*average_validation_data)

print('\n Accuracy on validation set : ' + accuracy_validation_data)

output_test_data = clf.predict(np.array(test_data))
result_test_data = (output_test_data == test_label)
result_float_test_data = result_test_data.astype(np.float)
average_test_data = np.average(result_float_test_data)
accuracy_test_data = str(100*average_test_data)

print('\n Accuracy on test set : ' + accuracy_test_data)