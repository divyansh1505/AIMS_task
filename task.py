import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

lap_filepath = 'data.csv'
lap_data = pd.read_csv(lap_filepath)

print(lap_data.columns)

y = lap_data['Price_euros']
print(y.head())

lap_features = ['Company', 'Product', 'Screen', 'IPSpanel', 'RetinaDisplay', 
                'CPU_company', 'CPU_freq', 'CPU_model', 'GPU_company', 'GPU_model']
X = lap_data[lap_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

def scoring(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)

object_cols = train_X.select_dtypes(include=['object']).columns.tolist()

print("Categorical variables:")
print(object_cols)

label_train_X = train_X.copy()
label_val_X = val_X.copy()

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  
label_train_X[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_val_X[object_cols] = ordinal_encoder.transform(val_X[object_cols])

print("MAE from Ordinal Encoding:") 
print(scoring(label_train_X, label_val_X, train_y, val_y))

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]), index=train_X.index)
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]), index=val_X.index)

num_train_X = train_X.drop(object_cols, axis=1)
num_val_X = val_X.drop(object_cols, axis=1)

OH_train_X = pd.concat([num_train_X.reset_index(drop=True), OH_cols_train.reset_index(drop=True)], axis=1)
OH_val_X = pd.concat([num_val_X.reset_index(drop=True), OH_cols_valid.reset_index(drop=True)], axis=1)

OH_train_X.columns = OH_train_X.columns.astype(str)
OH_val_X.columns = OH_val_X.columns.astype(str)

print("MAE from One-Hot Encoding:") 
print(scoring(OH_train_X, OH_val_X, train_y, val_y))
