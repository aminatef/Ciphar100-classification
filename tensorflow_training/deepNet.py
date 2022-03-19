from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

batch_size = 1000
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 300
optimizer = Adam( learning_rate=0.001)
validation_split = 0.15
verbosity = 1

layers_vs_acc_val=[]
layers_vs_acc_t=[]

layers_vs_acc_div2_node_t=[]
layers_vs_acc_div2_node_val=[]



def train(model):
    model_histroy = model.fit(input_train, y_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)
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


input_train = input_train.astype('float32').mean(axis=3)
input_test = input_test.astype('float32').mean(axis=3)

# Normalize data
input_train = input_train.reshape(50000,1024)/255
input_test = input_test.reshape(10000,1024)/255
norm = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
norm.adapt(input_train)
input_train = norm(input_train)
input_test  = norm(input_test)

#try and evaluate different depth and height of the network
for mul in range(10):
    t_list=list()
    val_list = list()
    for i in range(11):
        model = Sequential()
        model.add(layers.Dense(200*(mul+1), activation='relu',input_shape=(input_train.shape[1],)))
        for j in range(i):
            model.add(layers.Dense(200*(mul+1), activation='relu'))
        model.add(layers.Dense(no_classes, activation='softmax'))
        model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])
        file=f"{100*(mul+1)}_node/depth_{i+1}"
        val,t = train_and_evaluate(model,file)
        t_list.append(t)
        val_list.append(val)
        #print(f'layers number {len(model.layers)}')
        model.add(layers.Dense(no_classes, activation='softmax'))

    fig = plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('#Layers')
    plt.plot(np.arange(1,1+len(t_list)),t_list)
    plt.plot(np.arange(1,1+len(val_list)),val_list)
    plt.legend(["Training accuarcy", "Validation Accuracy"])
    plt.savefig(f"nodes{200*(mul+1)}_vs_layers")
    plt.close(fig)



# fl_nodes=2100
# for i in range(10):
#     model = Sequential()
#     model.add(layers.Dense(fl_nodes, activation='sigmoid',input_shape=(input_train.shape[1],)))
#     for j in range(i):
#         model.add(layers.Dense(fl_nodes-(j*200), activation='sigmoid'))
#     model.add(layers.Dense(no_classes, activation='softmax'))
#     model.compile(loss=loss_function,
#               optimizer=optimizer,
#               metrics=['accuracy'])
#     val,t = train_and_evaluate(model,f"div-2_layer/depth_{i+1}_div-2")
#     layers_vs_acc_div2_node_t.append(val)
#     layers_vs_acc_div2_node_val.append(t)
#     #print(f'layers number {len(model.layers)}')
#     model.add(layers.Dense(no_classes, activation='softmax'))

# fig = plt.figure()
# plt.ylabel('Accuracy')
# plt.xlabel('#Layers')
# plt.plot(np.arange(1,1+len(layers_vs_acc_div2_node_t)),layers_vs_acc_div2_node_t)
# plt.plot(np.arange(1,1+len(layers_vs_acc_div2_node_val)),layers_vs_acc_div2_node_val)
# plt.legend(["Training accuarcy", "Validation Accuracy"])
# plt.savefig("div-2NodeVsLayers")
# plt.close(fig)



   

# Generate generalization metrics
#score = model.evaluate(input_test, target_test, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')