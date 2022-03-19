from numpy.core.numeric import indices
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.losses import categorical_hinge


import matplotlib.pyplot as plt

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def custom_MCSVM_loss(y_true,y_pred):
    y_true=tf.cast(y_true,tf.int32)
    # d = tf.maximum(1+y_pred-y_true,0.0)

    idxs= tf.transpose(y_true)
    idxs= tf.reshape(idxs,(len(y_true),))
    row_indices = tf.range(tf.shape(idxs)[0])
    full_indices = tf.stack([row_indices, idxs], axis=1)
    yi = tf.gather_nd(y_pred, full_indices)

    #print("yi shape is: ",yi.shape)
    # print("y_pred shape is: ",y_pred.shape)
    # print("y_pred first 3 rows\n",y_pred.numpy()[:3])
    #print("y_true first 3 rows\n",y_true.numpy()[:3])
    
    yi=tf.reshape(yi,(len(yi),1))
    # print("yi_new shape is: ",yi.shape)
    # print("y_i first 3 rows\n",yi.numpy()[:3])
    diff = y_pred - yi
    # print("diff first 3 rows\n",diff.numpy()[:3])
    d = tf.maximum(1+ diff,0.0)

    arr = np.zeros(d.shape)
    arr[full_indices[:,0], full_indices[:,1]] = 1
    y_ones = tf.convert_to_tensor(arr,dtype=tf.float32)


    return tf.reduce_sum(tf.math.subtract(d,y_ones))
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

def train_evaluate(pca_len,node_num,x_train,x_test,y_train,y_test):
    no_epochs=5000
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    #Perform PCA
    pca = PCA(n_components=pca_len)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    x_val = x_train[-7500:]
    y_val = y_train[-7500:]
    x_train = x_train[:-7500]
    y_train = y_train[:-7500]

    y_train = y_train.astype("float32")
    y_test  = y_test.astype("float32")
    regularization = tf.keras.regularizers.L2(l2=0.01)
    model = keras.Sequential(
        [
            layers.Dense(node_num,
                kernel_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                activation="relu",
                kernel_regularizer=regularization,
                name="hidden_layer"),
                layers.Dense(100,
                kernel_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                activation="linear",
                name="output"),

        ]
    )
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=custom_MCSVM_loss,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                run_eagerly=True
                )
    history = model.fit(
    x_train,
    y_train,
    batch_size=10000,
    epochs=no_epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
    )
    
    history = history.history
    loss     = history["loss"]
    acc      = history["sparse_categorical_accuracy"]
    val_loss = history["val_loss"]
    val_acc  = history["val_sparse_categorical_accuracy"]

    max_val_acc = np.max(val_acc)
    max_training_acc = np.max(acc)

    fig = plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('#Epochs')
    plt.plot(np.arange(no_epochs),loss)
    plt.plot(np.arange(no_epochs),val_loss)
    plt.legend(["Training loss", "Validation loss"])
    plt.savefig(f"MCSVM/pca-{pca_len}_node-{node_num}_MCSVM_loss")
    plt.close(fig)

    fig = plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('#Epochs')
    plt.plot(np.arange(no_epochs),acc)
    plt.plot(np.arange(no_epochs),val_acc)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.savefig(f"MCSVM/pca-{pca_len}_node{node_num}_MCSVM_acc")
    plt.close(fig)

    model.evaluate(x_test,y_test)
    return max_val_acc,max_training_acc


x_train = x_train.astype("float32")
x_test  = x_test.astype("float32")

x_train = x_train.reshape(50000,3072)/255
x_test  = x_test.reshape(10000,3072)/255

pca=70
for i in range(10):
    t_acc=[]
    val_acc=[]
    for j in range(i+1):
        t_acc_1 ,val_acc_1 = train_evaluate(pca+(30*i),j*100,x_train,x_test,y_train,y_test)
        t_acc.append(t_acc_1)
        val_acc.append(val_acc_1)
    fig = plt.figure()
    plt.ylabel('acc')
    plt.xlabel('#nodes*10')
    plt.plot(np.arange(10),t_acc)
    plt.plot(np.arange(10),val_acc)
    plt.legend(["max Training acc", "max Validation acc"])
    plt.savefig(f"MCSVM/nodesVSacc_pca={pca+(30*i)}_MCSVM_acc")
    plt.close(fig)
    

# model.save("my_model")
# model = keras.models.load_model("my_model",compile=False)













