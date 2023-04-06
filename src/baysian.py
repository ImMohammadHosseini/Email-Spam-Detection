


import numpy as np
import pandas as pd

class Baysian():
    def __init__(self):
        self.vocab=[]
        self.p_ham=0.0
        self.p_spam=0.0
        self.p_spam_vocab={}
        self.p_ham_vocab={}
        
        
    def fit(self, data, train_data):
        
        counts=data['Prediction'].value_counts()
        self.vocab= train_data.columns[1:-1]
        
        self.p_ham = counts[0]/train_data.shape[0]
        self.p_spam = counts[1]/train_data.shape[0]
        
        Nspam = train_data.loc[data['Prediction'] == 1]
        Nspam = Nspam.drop(['Prediction', 'Email No.'], axis=1)

        Nham = train_data.loc[data['Prediction'] == 0]
        Nham = Nham.drop(['Prediction', 'Email No.'], axis=1)
        
        spam_word_num = Nspam.apply(len).sum()
        ham_word_num =  Nham.apply(len).sum()
        
        spam_vocab_num = Nspam.sum()
        ham_vocab_num = Nham.sum()
        
        self.p_spam_vocab= {key:((v+1)/(spam_word_num+len(self.vocab))) for key,v in zip(self.vocab, spam_vocab_num)}
        self.p_ham_vocab= {key:((v+1)/(ham_word_num+len(self.vocab))) for key,v in zip(self.vocab, ham_vocab_num)}
        
        
        
    def test(self, test_data):
        y_hat=[]
        
        for i, row in test_data.iterrows():
            spam = np.log(self.p_spam)
            ham = np.log(self.p_ham)
            sum_ham=0
            sum_spam=0
            for j,j1 in enumerate(test_data):
                #if test_data.loc[i, j1] != 0:
                sum_spam += np.log(self.p_spam_vocab.get(test_data.columns[j]))*test_data.loc[i, j1]
                sum_ham += np.log(self.p_ham_vocab.get(test_data.columns[j]))*test_data.loc[i, j1]
                    #spam = np.log(spam * self.p_spam_vocab.get(test_data.columns[j])**test_data.loc[i, j1])
                    #ham = np.log(ham * self.p_ham_vocab.get(test_data.columns[j])**test_data.loc[i, j1])
            spam += sum_spam
            ham += sum_ham
            #print(spam)
            #print(ham)
            if spam <= ham:
                y_hat.append(0)
            else:
                y_hat.append(1)
                
        
        return y_hat
    def accuracy(self, y_test, y_hat):
        cmp=[True if y_test[i]==y_hat[i] else False for i in range(len(y_test))]
        print ('Accuracy:%f' %(cmp.count(True)/len(y_test)))
 