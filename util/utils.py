from tsne import bh_sne
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools

def plot_latent_space(X, y, file_name):
    """
    This function employs TSNE to convert the latent space to 2D space and plots the result.
    """
    X = tf.cast(X, dtype='float64')
    X = bh_sne(X)
    walking = X[y==1]
    walking_up = X[y==2]
    walking_down = X[y==3]
    sitting = X[y==4]
    standing = X[y==5]
    laying = X[y==6]
    
    colors = ['r', 'c', 'k', 'y', 'm', 'g']
    plt.figure(figsize=(12, 10))
    WALKING = plt.scatter(walking[:, 0], walking[:, 1], marker='x', color=colors[0], alpha=0.3)
    WALKING_UPSTAIRS = plt.scatter(walking_up[:, 0], walking_up[:, 1], marker='+', color=colors[1], alpha=0.3)
    WALKING_DOWNSTAIRS  = plt.scatter(walking_down[:, 0], walking_down[:, 1], marker='^', color=colors[2], alpha=0.3)
    SITTING  = plt.scatter(sitting[:, 0], sitting[:, 1], marker='o', color=colors[3], alpha=0.3)
    STANDING  = plt.scatter(standing[:, 0], standing[:, 1], marker='o', color=colors[4], alpha=0.3)
    LAYING = plt.scatter(laying[:, 0], laying[:, 1], marker='o', color=colors[5], alpha=0.3)

    plt.legend((WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING),
               ('WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.savefig(file_name+'.png')
    plt.show()

def plot_confusion_matrix(model, X, y, class_names, file_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    y_pred = model.predict(X)
    # Compute confusion matrix
    cnf_matrix  = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(12, 10))

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(file_name+'.png')
    plt.show()

def print_result(model, X_train, y_train, X_test, y_test):
    """
    This function print the accuracy of the model.
    """
    print('Train accuracy: ', model.score(X_train, y_train))
    print('Test accuracy: ', model.score(X_test, y_test))

def rf(X_train, y_train, n_estimators=100):
    """
    This function builds a random forest.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf