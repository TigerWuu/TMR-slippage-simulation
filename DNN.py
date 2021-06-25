import numpy as np

class DNN():
    def __init__(self,shape):
        self.shape=shape
        self.w=[]
        self.w_delta=[]
        for i in range(len(shape)-1):
            self.w.append([])
            self.w_delta.append([])
            self.w[i]=np.random.randn(shape[i],shape[i+1])
            self.w_delta[i]=np.zeros([shape[i],shape[i+1]])
        self.w=np.array(self.w)
        self.w_delta=np.array(self.w_delta)
        self.b=[]
        self.b_delta=[]
        self.v=[]
        self.z=[]
        for i in range(len(shape)-1):
            self.b.append([])
            self.b_delta.append([])
            self.v.append([])
            self.b[i]=np.random.randn(shape[i+1])
            self.b_delta[i]=np.zeros(shape[i+1])
            self.v[i]=np.zeros([shape[i+1]])
        for i in range(len(shape)):
            self.z.append([])
            self.z[i]=np.zeros([shape[i]])
        self.b=np.array(self.b)
        self.b_delta=np.array(self.b_delta)
        self.v=np.array(self.v)
        self.z=np.array(self.z)
        
    def predict(self,x):
        x = np.array(x)
        self.z[0]=x.reshape(self.z[0].shape)

        for i in range(len(self.shape)-1):
            self.v[i]=np.dot(self.z[i],self.w[i])+self.b[i]

            self.z[i+1]=self.sigmoid(self.v[i])
        return self.z[i+1]

    def gradient(self,y):
        for i in range(len(self.shape)-2,-1,-1):
            if i == (len(self.shape)-2):
                delta = (self.sigmoid(self.v[-1])-y.reshape(self.v[-1].shape))*self.sigmoid_p(self.v[-1])
            else:
                delta = np.dot(self.w[i+1],delta) * self.sigmoid_p(self.v[i])
            self.b_delta[i] = delta
            self.w_delta[i] = np.dot(self.z[i].reshape((len(self.z[i]),1)),delta.reshape(1,len(delta)))   
    
    def training(self,x, y, epoch, alpha):
        train_x = x
        train_y = y
        avg_del_b = 0
        avg_del_w = 0
        for i in range(epoch):
            print("epoch : ", i)
            for j in range(len(train_x)):
                self.predict(train_x[j])
                self.gradient(train_y[j])
                avg_del_b += 1/len(train_x) * self.b_delta
                avg_del_w += 1/len(train_x) * self.w_delta
            self.b -= alpha*avg_del_b
            self.w -= alpha*avg_del_w
            avg_del_b=0
            avg_del_w=0

    
    def sigmoid(self,x):
        sig = 1/(1+np.exp(-x))
        return sig
        
    def sigmoid_p(self,x):
        sig_p = self.sigmoid(x)*(1-self.sigmoid(x))
        return sig_p

if __name__ == "__main__":
    pass