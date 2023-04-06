
import pandas as pd
import numpy as np
from src.baysian import Baysian
from src.ensemble import Ensemble

if __name__ == "__main__":
    data = pd.read_csv('data/emails.csv')


    data= data.sample(frac=1)
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    #labels= list(data['Prediction'])

    y_test = list(test['Prediction'])
    test = test.drop(['Prediction', 'Email No.'], axis=1)

    baysian = Baysian()
    baysian.fit(data, train)
    y_hat = baysian.test(test)
    baysian.accuracy(y_test, y_hat)

    ensembleModel = Ensemble(20, 100)
    ensembleModel.fit(data)
    y_hat= ensembleModel.test(60)