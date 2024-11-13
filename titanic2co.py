# Importación de librerías
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

# Carga de datos
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

# Preparación de los datos
titanic_train.set_index('PassengerId', inplace=True)
titanic_test.set_index('PassengerId', inplace=True)

# Imputación de valores faltantes
titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode()[0], inplace=True)
titanic_train['EdadDesconocida'] = titanic_train['Age'].isnull()
titanic_test['EdadDesconocida'] = titanic_test['Age'].isnull()
titanic_train['Age'].fillna(titanic_train['Age'].median(), inplace=True)
titanic_test['Age'].fillna(titanic_train['Age'].median(), inplace=True)
titanic_test['Fare'].fillna(titanic_train['Fare'].median(), inplace=True)
titanic_train.drop(['Cabin'], axis=1, inplace=True)
titanic_test.drop(['Cabin'], axis=1, inplace=True)
titanic_train.drop(['Name', 'Ticket'], axis=1, inplace=True)
titanic_test.drop(['Name', 'Ticket'], axis=1, inplace=True)
titanic_train['ViajaSolo'] = ((titanic_train['SibSp'] + titanic_train['Parch']) == 0)
titanic_test['ViajaSolo'] = ((titanic_test['SibSp'] + titanic_test['Parch']) == 0)
titanic_train['MenorDeEdad'] = (titanic_train['Age'] < 18.0)
titanic_test['MenorDeEdad'] = (titanic_test['Age'] < 18.0)
titanic_train['AdultoMayor'] = (titanic_train['Age'] > 55.0)
titanic_test['AdultoMayor'] = (titanic_test['Age'] > 55.0)
titanic_train['EsMujer'] = (titanic_train['Sex'] == 'female')
titanic_test['EsMujer'] = (titanic_test['Sex'] == 'female')
titanic_train.drop(['Sex'], axis=1, inplace=True)
titanic_test.drop(['Sex'], axis=1, inplace=True)
cols_categoricas = ['Pclass', 'Embarked']
titanic_train = pd.get_dummies(titanic_train, columns=cols_categoricas)
titanic_test = pd.get_dummies(titanic_test, columns=cols_categoricas)

# Separación de conjunto para validación
X_train_val = titanic_train.drop('Survived', axis=1)
y_train_val = titanic_train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)

# Evaluación de modelos
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

# Entrenamiento y evaluación de AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train, y_train)
print("AdaBoost Performance:")
evaluate_model(ada, X_val, y_val)

# Entrenamiento y evaluación de Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=0)
gb.fit(X_train, y_train)
print("\nGradient Boosting Performance:")
evaluate_model(gb, X_val, y_val)

# Entrenamiento y evaluación de XGBoost
xgb_model = XGBClassifier(n_estimators=100, random_state=0)
xgb_model.fit(X_train, y_train)
print("\nXGBoost Performance:")
evaluate_model(xgb_model, X_val, y_val)

# Entrenamiento y evaluación de Stacking
estimators = [('rf', RandomForestClassifier(n_estimators=100, random_state=0)), ('ada', AdaBoostClassifier(n_estimators=100, random_state=0)), ('gb', GradientBoostingClassifier(n_estimators=100, random_state=0))]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)
print("\nStacking Performance:")
evaluate_model(stacking, X_val, y_val)
