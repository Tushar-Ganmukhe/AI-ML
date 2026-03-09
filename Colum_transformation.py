import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

df=pd.read_csv('D:\Full_Restart\AI & ML\data\covid_toy.csv')

print(df)

x_train,X_test,y_train,y_test=train_test_split(df.drop(columns='has_covid'),df['has_covid'],test_size=0.2,random_state=42)

#-----------------------------  Colum_Transformation --------------------------------------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer

tf=ColumnTransformer(transformers=[

('tf1',SimpleImputer(),['fever']),
('tf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
('tf3',OneHotEncoder(drop='first',sparse_output=False),['gender','city'])],

remainder='passthrough',sparse_threshold=0,verbose_feature_names_out=False)#for gat alway np arry not sparse we nedd seat threshold=0 (dinsity =no. of non-zero valu/ total value)

tranformed=tf.fit_transform(x_train)

print(tranformed)
df1 = pd.DataFrame(
    tranformed,
    columns=tf.get_feature_names_out()
)

print(df1.astype('int'))

final_df=df1.astype('int')

final_df.to_excel("transformed_data.xlsx", index=False)