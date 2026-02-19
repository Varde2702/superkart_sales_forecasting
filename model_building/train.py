import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib, os

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

X_train = train_df.drop('Product_Store_Sales_Total', axis=1)
y_train = train_df['Product_Store_Sales_Total']

X_test = test_df.drop('Product_Store_Sales_Total', axis=1)
y_test = test_df['Product_Store_Sales_Total']

model = XGBRegressor(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(model, params, cv=3, scoring='r2')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
preds = best_model.predict(X_test)

print('R2:', r2_score(y_test, preds))
print('RMSE:', mean_squared_error(y_test, preds, squared=False))

os.makedirs('../deployment', exist_ok=True)
joblib.dump(best_model, '../deployment/best_model.pkl')
