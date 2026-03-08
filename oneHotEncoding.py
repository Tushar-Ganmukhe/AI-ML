import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OneHotEncoder

from My_Function import show_encoded_data as sh

df=pd.read_csv('data\cars.csv')

#------- oneHot encoding using pandas----note(it not store position of encodeaded colume so  not use in build ML models)

encoded=pd.get_dummies(df,columns=['fuel']) # This give Output in True/False insted of 1/0 to fix use (dtyp=int) parameter
encoded1=pd.get_dummies(df,columns=['fuel'],dtype=int)

#but to fix multicolenarity issu we nedd to remove one colum (n-1) colum it

encoded2=pd.get_dummies(df,columns=['fuel','owner'],dtype=int,drop_first=True)

#--print(encoded2)

#------------------------------------------Using Sklearn-------------------------------------------

x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=42)
#train_test_split(X, y, test_size=0.2, random_state=0)--> x=input,y=output(pridaction), ramdom_state=seed(Pattern to split data)

#--print(x_train)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(drop='first',sparse_output=False) # sparse false to conver defolt sparse matrix to nparry

x_train_new=ohe.fit_transform(x_train[['fuel']])

print(x_train_new)
sh(x_train_new,ohe)



#Note----------------------------------------------------------------------------------------------------

# 1)-> print(df.iloc[0:2,0:4]) #iloc[row,colum]

#2)-> Multicollinearity in Machine Learning
#Multicollinearity means two or more independent variables (features) are highly correlated with each other.
#Because of this, the model cannot clearly understand which variable is actually affecting the output''''

# 3)-> onehotencoder(): 1)it bdefult output is sparce matrix we nedd to conver into np arry
#                       2) it only get attribut in 2D Arry [['name_of_Colum]]