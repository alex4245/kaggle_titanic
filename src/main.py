import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
RESULT_PATH = 'data/result.csv'


class NameAttributeConv(TransformerMixin):
    def __init__(self, name_column='Name'):
        self._name_column = name_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        column = X[self._name_column].apply(lambda x: re.search(r'\w+\.', x).group())
        X[self._name_column] = column
        return X


train_data = pd.read_csv(TRAIN_DATA_PATH)
train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=42)

proc_train_set = train_set.drop(["PassengerId", "Survived", "Ticket", "Cabin"], axis=1)
survived = train_set["Survived"].copy()

num_attr = ["Age", "Fare"]
cat_attr = ["Sex", "Embarked", "Pclass", "Name"]

cat_transformer_categories = [
    ['male', 'female'],
    ['S', 'C', 'Q'],
    [1, 2, 3],
    ['Mr.', 'Miss.', 'Major.', 'Mrs.', 'Master.', 'Rev.', 'Dr.', 'Col.',
    'Mlle.', 'Capt.', 'Mme.', 'Ms.', 'Countess.', 'Lady.']
] 

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

cat_transformer = Pipeline([
    ('name_attr_conv', NameAttributeConv(name_column='Name')),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('cat', OneHotEncoder(handle_unknown='ignore', categories=cat_transformer_categories)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_attr),
        ('cat', cat_transformer, cat_attr),
    ]
)
prepare = preprocessor.fit_transform(proc_train_set)
rfr = RandomForestRegressor(random_state=1, max_depth=5)
model = rfr.fit(prepare, survived)

test_set = pd.read_csv(TEST_DATA_PATH)
test_passenger_ids = test_set["PassengerId"].copy()
prepare_test_set = preprocessor.fit_transform(test_set)
predict_values = model.predict(prepare_test_set)

result = pd.DataFrame(test_passenger_ids)
result["Survived"] = np.around(predict_values).astype(int)
result.to_csv(RESULT_PATH, index=False)
