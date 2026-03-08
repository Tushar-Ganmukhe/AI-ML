#----------------------------------- Noraml Apporach -----------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('data/covid_toy.csv')
print(df.head)

x_train,x_test,y_train,y_test= train_test_split(df.drop(columns='has_covid'),df['has_covid'],test_size=0.2,random_state=42)

print('----------------------------------------------------------------------------------------------------')
print(y_train)

print('----------------------------------------------------------------------------------------------------')
print(y_test)
print('----------------------------------------------------------------------------------------------------')

ohe=OneHotEncoder(drop='first',sparse_output=False)

x_train_gender=ohe.fit_transform(x_train[['gender']])
print(x_train_gender.shape)

print('----------------------------------------------------------------------------------------------------')
x_test_gender=ohe.transform(x_test[['gender']])
print('----------------------------------------------------------------------------------------------------')

print('----------------------------------------------------------------------------------------------------')

x_train_city= ohe.fit_transform(x_train[['city']])
print(x_train_city.shape)
x_test_city=ohe.transform(x_test[['city']])
print(x_test_city.shape)
print('----------------------------------------------------------------------------------------------------')


#-----------------SimpleImputer---------------------------------------------------------

from sklearn.impute import SimpleImputer

si=SimpleImputer()

x_train_fever=si.fit_transform(x_train[['fever']])


fever=pd.DataFrame(x_train_fever,columns=si.get_feature_names_out())
print(fever)

x_test_fever=si.transform(x_test[['fever']])

fevert=pd.DataFrame(x_test_fever,columns=si.get_feature_names_out())
print(fevert.shape)




#-----------------ordinalEncoder---------------------------------------------------------

from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder(categories=[['Mild','Strong']]) # low pariority to high

x_train_cough=oe.fit_transform(x_train[['cough']])
print(x_train_cough.shape)
x_test_cougf=oe.transform(x_test[['cough']])
print(x_test_cougf.shape)

#-----------------------Age Extraction-------------------------------------------

x_train_age = x_train.drop(columns=['gender','fever','cough','city']).values # valuse use foe converting it in to np arry
print(x_train_age.shape)

x_test_age = x_test.drop(columns=['gender','fever','cough','city']).values
print(x_test_age.shape)

#--------------------------Contacnate----------------------------------------

x_train_transform=np.concatenate((x_train_gender,x_train_city,x_train_fever,x_train_cough,x_train_age),axis=1)

print(x_train_transform.shape)