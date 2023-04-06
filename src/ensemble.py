
import numpy as np
import pandas as pd
from .baysian import Baysian

class Ensemble():
    def __init__(self, machine_num, tdata_num):
        self.machine_num= machine_num
        self.tdata_num = tdata_num
        self.data=pd.DataFrame()
        self.sub_datas=[]
        self.models=[]
        
    def bootstraping(self):
        np.random.seed(1)
        randint=[[np.random.randint(0,5172) for j in range(self.tdata_num)]for i in range(self.machine_num)]
        self.sub_datas= [self.data.loc[randint[0],:] for _ in range(self.machine_num)]
        
    def fit(self, data):
        self.data= data
        self.bootstraping()
        
        self.models= [Baysian() for _ in range(self.machine_num)]
        
        for i in range(self.machine_num):
            self.models[i].fit(self.data, self.sub_datas[i])
    
    def test(self, data_num):
        rand_for_test= [np.random.randint(0,5172) for _ in range(data_num)]
        test_data= self.data.loc[rand_for_test,:]
        
        y_test = list(test_data['Prediction'])
        test = test_data.drop(['Prediction', 'Email No.'], axis=1)
        
        y_hats=[]
        for i in range(self.machine_num):
            y_hats.append(self.models[i].test(test))
            
        y_hat=[]
        for i in range(data_num):
            ones=0
            zeros=0
            for j in range(self.machine_num):
                if y_hats[j][i] == 1:
                    ones+=1
                elif y_hats[j][i] == 0:
                    zeros+=1
            if ones >= zeros:
                y_hat.append(1)
            elif zeros > ones:
                y_hat.append(0)
             
        self.accuracy(y_test, y_hat)
        
        return y_hat
        
    def accuracy(self, y_test, y_hat):
        cmp=[True if y_test[i]==y_hat[i] else False for i in range(len(y_test))]
        print ('Accuracy:%f' %(cmp.count(True)/len(y_test)))
