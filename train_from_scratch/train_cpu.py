import cv2
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar_data():
    data_dict = unpickle("cifar-100-python/train")
    labels = unpickle("cifar-100-python/meta")
    fine_labels = labels[b'fine_label_names']

    features = np.array(data_dict[b'data']).reshape(50000,3,32,32).transpose(0,2,3,1)

    grey_scale_imgs = np.zeros((50000,32,32))
    for i in range(features.shape[0]):
        grey_scale_imgs[i]= cv2.cvtColor(features[i], cv2.COLOR_BGR2GRAY)
        #grey_scale_imgs[i] = cv2.normalize(grey_scale_imgs[i],None, 0, 1, cv2.NORM_MINMAX)


    training_labels = np.array(data_dict[b'fine_labels'])
    return(np.c_[grey_scale_imgs.reshape(50000,1024),np.ones(50000)],training_labels,fine_labels)

def load_cifar_test():
    data_dict = unpickle("cifar-100-python/test")
    labels = unpickle("cifar-100-python/meta")
    fine_labels = labels[b'fine_label_names']
    features = np.array(data_dict[b'data']).reshape(10000,3,32,32).transpose(0,2,3,1)
    grey_scale_imgs = np.zeros((10000,32,32))
    for i in range(features.shape[0]):
        grey_scale_imgs[i]= cv2.cvtColor(features[i], cv2.COLOR_BGR2GRAY)
    training_labels = np.array(data_dict[b'fine_labels'])
    return(np.c_[grey_scale_imgs.reshape(10000,1024),np.ones(10000)],training_labels,fine_labels)


class MCSVM:
    def __init__(self,training_data,training_labels,labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.labels = labels
        self.weights= np.random.normal(scale = 0.7,size = (len(self.labels),training_data[0].shape[0]))
        self.input_length = training_data[0].shape[0]
        self.data_length = len(training_data)

    def test(self):
         test1 = np.array([0,1,2,3,4,5])

    def train(self,alpha,epsilon):
        saved = 0
        scores = np.zeros(len(self.labels))
        print("scores shape :{}".format(scores.shape))
        step = 0
        lo=0
        grad = np.zeros((len(self.labels),self.input_length))
        print("grad shape :{}".format(grad.shape))
        while(np.linalg.norm(grad)) > epsilon or step == 0 or True:
            grad = np.zeros((len(self.labels),self.input_length))
            lo=0
            step+=1
            loss_i=0
            for i in range(self.data_length):
                loss_i=0
                #getting a training sample
                training_sample = self.training_data[i]

                #calculating (Wr.Xi) for r in range(100)
                scores = np.dot(self.weights,training_sample)

                #index of max score
                pred_i = np.argmax(scores)

                #true label of training sample
                true_pred_i = self.training_labels[i]

                

                if(pred_i != true_pred_i):
                    d = np.maximum(1 + scores - scores[true_pred_i], 0)
                    d[true_pred_i] = 0
                    d = d > 0
                    #if r == y_i 
                    grad[true_pred_i]-=training_sample*np.sum(d)

                    idx = np.where(d==1)
                    #idx = [indexs where (1+Wj.Xi-Wy_i.Xi)>0 && j !=y_i]
                    grad[idx]+=training_sample
                    loss_i=np.sum(d)
                lo+=loss_i  

            grad = grad*(alpha/self.data_length)
            self.weights -= grad
            if (step%100) == 0:
                saved+=1
                np.savetxt('data_testpy{}_alpha{}.csv'.format(saved,alpha), self.weights, delimiter=',')
            print("Loss #{}: {}, delta: {}".format(step, lo, np.linalg.norm(grad)))
            

if __name__ == "__main__": 
    data = load_cifar_data()
        # print(data[0][0].shape)
        # print(data[0][156][1024])
        # final = cv2.normalize(data[0][0],None, 0, 1, cv2.NORM_MINMAX)
        # cv2.imwrite("data.png",final.reshape(32,32))
    m = MCSVM(data[0],data[1],data[2])
    m.train(0.001,0.1)

            





