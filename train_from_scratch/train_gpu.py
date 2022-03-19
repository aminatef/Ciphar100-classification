import cv2
import numpy as np
from numba import jit,cuda,guvectorize,u1, i8,f4
import math
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

@cuda.jit
def calc_scores(training_data,weights,scores):
    tx,ty = cuda.grid(2)
    if tx >=  training_data.shape[1] or ty >= training_data.shape[0]:
        return
    temp=0
    for i in range(1025):
        temp+=training_data[ty][i] * weights[tx][i]
    scores[ty][tx] = temp



@cuda.jit
def get_missClassifications(scores,res_max_idx,loss,delta,training_labels):
    #scores.shape = (batch_size*,100) contains the scores of every training sample in a batch
    #loss.shape = (batch_size,1) contains loss of each training sample
    #delta.shape = (batch_size,1) contains delta(number of missclassification) of each training sample
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx+ty*bw
    max_num = -1
    max_idx = 0
    if idx>=scores.shape[0]:
        return
    loss[training_labels[idx]] = 0
    for i in range(scores.shape[1]):
        if(scores[idx][i]>max_num):
            max_num=scores[idx][i]
            max_idx=i
        temp = 1+scores[idx][i]-scores[idx][training_labels[idx]]
        if(temp > 0 and i != training_labels[idx]):
            loss[idx] += temp
            delta[idx]+=1
    res_max_idx[idx] = max_idx



@cuda.jit
def calc_grad(scores,training_labels,res_max_idx,delta,grad):

    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx+ty*bw
    if idx>=scores.shape[0]:
        return

    if training_labels[idx]!=res_max_idx[idx]:
        grad[idx][training_labels[idx]] = (-1*delta[idx])
        for i in range(scores.shape[1]):
            temp = 1+scores[idx][i]-scores[idx][training_labels[idx]]
            if(temp>0 and i !=training_labels[idx] ):
                grad[idx][i] = 1
@cuda.jit
def empty_grad(grad,delta,loss):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx+ty*bw
    y = cuda.blockIdx.y
    if idx>=grad.shape[0]:
        return
    grad[idx][y] = 0
    delta[idx]=0
    loss[idx]=0

        






    


class MCSVM:
    def __init__(self,training_data,training_labels,labels):
        self.training_data   = training_data
        self.training_labels = training_labels
        self.labels          = labels
        self.weights         = np.random.normal(scale = 0.7,size = (len(self.labels),training_data[0].shape[0]))
        self.input_length    = training_data[0].shape[0]
        self.data_length     = len(training_data)
        self.batch_size      = 25000

    

    def test(self):
        scores = np.arange(9).reshape(3,3)
        res = np.zeros((3,1))
        d_arr_scroes  = cuda.to_device(scores)
        d_arr_res   = cuda.to_device(res)
        threadsperblock = (1000,1)
        blockspergrid = (25,1)
        get_max[blockspergrid, threadsperblock](d_arr_scroes,d_arr_res)

    def train(self,alpha,epsilon):
        
        grad = np.zeros((self.batch_size,len(self.labels)))
        scores = np.zeros((self.batch_size,len(self.labels)))
        res_max_idx = np.zeros(self.batch_size)
        loss_batch = np.zeros(self.batch_size)
        delta_batch = np.zeros(self.batch_size)

        saved = 0
        step  = 0
        lo    = 0

        threadsperblock_train_gpu = (100, 10)
        blockspergrid_x = math.ceil(self.training_data.shape[1] / threadsperblock_train_gpu[0])
        blockspergrid_y = math.ceil(self.training_data.shape[0] / threadsperblock_train_gpu[1])
        blockspergrid_train_gpu = (blockspergrid_x, blockspergrid_y)

        threadsperblock_grad = (1000,1)
        blockspergrid_grad_x = math.ceil(self.batch_size / threadsperblock_grad[0])
        blockspergrid_grad_y = 1
        blockspergrid_grad   = (blockspergrid_grad_x,blockspergrid_grad_y)

        threadsperblock_empty_grad = (1000,1)
        blockspergrid_grad_x_e = math.ceil(self.batch_size / threadsperblock_empty_grad[0])
        blockspergrid_grad_y_e = 100
        blockspergrid_empty_grad   = (blockspergrid_grad_x_e,blockspergrid_grad_y_e)



        num_of_batches = (50000//self.batch_size)
        batch_length = (50000//num_of_batches)

        d_arr_list_training =  [ cuda.to_device(self.training_data[i*batch_length:50000//(num_of_batches-i)]) for i in range(num_of_batches)]
        d_arr_list_labels   =  [ cuda.to_device(self.training_labels[i*batch_length:50000//(num_of_batches-i)]) for i in range(num_of_batches)]
        
        d_arr_scroes            = cuda.to_device(scores)
        d_arr_res_max_idx       = cuda.to_device(res_max_idx)
        d_arr_loss_batch        = cuda.to_device(loss_batch)
        d_arr_delta_batch       = cuda.to_device(delta_batch)
        d_arr_grad              = cuda.to_device(grad)
        grad_1 = np.zeros((self.batch_size,len(self.labels)))
        d_arr_grad    = cuda.to_device(grad_1)
        d_arr_weights = cuda.to_device(self.weights)

        print("grad shape :{}".format(grad.shape))
        while(np.linalg.norm(grad)) > epsilon or step == 0 or True:
            grad   = np.zeros((self.batch_size,len(self.labels)))  
            loss = np.zeros(self.batch_size)
            

            lo=0
            step+=1
            loss_i=0
            for i in range(num_of_batches):
                #grad_1 = np.zeros((self.batch_size,len(self.labels)))
                #
                calc_scores[blockspergrid_train_gpu, threadsperblock_train_gpu](d_arr_list_training[i],
                                                                                d_arr_weights,
                                                                                d_arr_scroes)

                get_missClassifications[blockspergrid_grad, threadsperblock_grad](d_arr_scroes,
                                                                              d_arr_res_max_idx,
                                                                              d_arr_loss_batch,
                                                                              d_arr_delta_batch,
                                                                              d_arr_list_labels[i])

                calc_grad[blockspergrid_grad, threadsperblock_grad](d_arr_scroes,
                                                                d_arr_list_labels[i],
                                                                d_arr_res_max_idx,
                                                                d_arr_delta_batch,
                                                                d_arr_grad)

                empty_grad[blockspergrid_empty_grad,threadsperblock_empty_grad](d_arr_grad,d_arr_delta_batch,d_arr_loss_batch)



                #for j in range(batch_length):
                    #temp = self.weights
                    #self.weights -= alpha*np.matmul(grad_1[j].reshape(100,1),self.training_data[i*batch_length+j].reshape(1,1025))
            print("Loss #{}: {}, delta: {}".format(step, lo, np.linalg.norm(self.weights)))
            if (step%100) == 0:
                saved+=1
                np.savetxt('data_testpy{}_alpha{}.csv'.format(saved,alpha), self.weights, delimiter=',')
            


                

            # for i in range(self.data_length):
            #     loss_i=0
            #     #getting a training sample
            #     training_sample = self.training_data[i]

            #     #calculating (Wr.Xi) for r in range(100)
            #     scores = np.dot(self.weights,training_sample)

            #     #index of max score
            #     pred_i = np.argmax(scores)

            #     #true label of training sample
            #     true_pred_i = self.training_labels[i]

                

            #     if(pred_i != true_pred_i):
            #         d = np.maximum(1 + scores - scores[true_pred_i], 0)
            #         d[true_pred_i] = 0
            #         d = d > 0
            #         #if r == y_i 
            #         grad[true_pred_i]-=training_sample*np.sum(d)

            #         idx = np.where(d==1)
            #         #idx = [indexs where (1+Wj.Xi-Wy_i.Xi)>0 && j !=y_i]
            #         grad[idx]+=training_sample
            #         loss_i=np.sum(d)
            #     lo+=loss_i  

            

     
if __name__ == "__main__":
    data = load_cifar_data()
    # print(data[0][0].shape)
    # print(data[0][156][1024])
    # final = cv2.normalize(data[0][0],None, 0, 1, cv2.NORM_MINMAX)
    # cv2.imwrite("data.png",final.reshape(32,32))
    m = MCSVM(data[0],data[1],data[2])
    m.train(0.0001,0.1)

            
