from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout
from tensorflow.keras.layers import MaxPooling2D,BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

batch_size = 64
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 500
optimizer = Adam( learning_rate=0.001)
validation_split = 0.15
verbosity = 1

layers_vs_acc_val=[]
layers_vs_acc_t=[]

layers_vs_acc_div2_node_t=[]
layers_vs_acc_div2_node_val=[]



def train(model):
    aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

    model_histroy = model.fit(
        input_train, y_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)
    model.evaluate(input_test,y_test)
    return model_histroy.history

def train_and_evaluate(model,file_name):
    histroy  = train(model)
    loss     = histroy["loss"]
    acc      = histroy["accuracy"]
    val_loss = histroy["val_loss"]
    val_acc  = histroy["val_accuracy"]

    max_acc_idx = np.argmin(val_loss)
    max_val_acc = val_acc[max_acc_idx]
    training_acc = acc[max_acc_idx]
    
    fig = plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('#Epochs')
    plt.plot(np.arange(no_epochs),loss)
    plt.plot(np.arange(no_epochs),val_loss)
    plt.legend(["Training loss", "Validation loss"])
    plt.savefig(file_name+"_loss")
    plt.close(fig)

    fig = plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('#Epochs')
    plt.plot(np.arange(no_epochs),acc)
    plt.plot(np.arange(no_epochs),val_acc)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.savefig(file_name+"_acc")
    plt.close(fig)

    return max_val_acc,training_acc

(input_train, y_train), (input_test, y_test) = cifar100.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)
input_train = input_train.astype('float32')/255
input_test = input_test.astype('float32')/255

# Normalize data
norm = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
norm.adapt(input_train)
input_train = norm(input_train)
input_test  = norm(input_test)


# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(BatchNormalization())

# model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

# model.add(Flatten())
# regularization = tf.keras.regularizers.L2(l2=0.001)
# model.add(Dense(2048,kernel_regularizer=regularization, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(2048,kernel_regularizer=regularization, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(100, activation='softmax'))
# model.summary()
t_acc=[]
val_acc=[]
for i in range(5,6):
    model = Sequential()
    for j in range(i+1):
        model.add(Conv2D(32+(j*32), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32+(j*32), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.AveragePooling2D((2, 2), padding="same"))
        model.add(Dropout(0.2))
    model.add(Flatten())
    regularization = tf.keras.regularizers.L2(l2=0.001)
    model.add(Dense(2048,kernel_regularizer=regularization, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2048,kernel_regularizer=regularization, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(100, activation='softmax'))
    
    opt = RMSprop(lr=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    x,y=train_and_evaluate(model,f"CONV_TEST_FOR/conv2d_VGG_conv={i+1}")
    val_acc.append(x)
    t_acc.append(y)

fig = plt.figure()
plt.ylabel('acc')
plt.xlabel('#convLayers')
plt.plot(np.arange(5),t_acc)
plt.plot(np.arange(5),val_acc)
plt.legend(["training acc", "validation acc"])
plt.savefig(f"CONV_TEST_FOR/convLayers")
plt.close(fig)
    


# compile model
# opt = RMSprop(lr=0.001)
# model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train_and_evaluate(model,"conv2d_VGG")
