import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Reading data
dataset = pd.read_excel('FilePath')

# 检测数据集中是否存在空缺值
dataset.dropna(inplace=True)

# Detects whether there is a gap value in the data set
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 7].values

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Data standardization and feature scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy_loss(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


# Define to calculate the weighted cross entropy loss
def weighted_cross_entropy_loss(y_true, y_pred, class_weights):
    return log_loss(y_true, y_pred, sample_weight=class_weights[y_true])


# Calculated class weight
class_weights = np.bincount(y_train)
class_weights = 1. / class_weights
class_weights = class_weights / class_weights.sum()


# evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    ce_loss = log_loss(y_test, y_pred_proba)
    wce_loss = weighted_cross_entropy_loss(y_test, y_pred_proba, class_weights)

    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    y_pred_proba_tensor = torch.tensor(y_pred_proba, dtype=torch.float32)
    focal_loss = FocalLoss()(y_pred_proba_tensor, y_test_tensor).item()

    return accuracy, ce_loss, wce_loss, focal_loss, y_pred


# Definition model list
models = {
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic': LogisticRegression(solver='sag', multi_class='multinomial', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=42),
    'GBDT': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    # 'MLP': MLPClassifier(hidden_layer_sizes=(5, 2), max_iter=10000, random_state=42),
}

# Evaluate the model's performance on the test set
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy, ce_loss, wce_loss, focal_loss, y_pred = evaluate_model(model, X_test, y_test)
    print(f'\n{name} - Test Set Results')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Cross-Entropy Loss: {ce_loss:.4f}')
    print(f'Weighted Cross-Entropy Loss: {wce_loss:.4f}')
    print(f'Focal Loss: {focal_loss:.4f}')
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

# 10-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    all_val_predictions = []
    all_val_targets = []
    all_ce_losses = []
    all_wce_losses = []
    all_focal_losses = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model.fit(X_train_fold, y_train_fold)
        _, ce_loss, wce_loss, focal_loss, val_pred = evaluate_model(model, X_val_fold, y_val_fold)

        all_val_predictions.extend(val_pred)
        all_val_targets.extend(y_val_fold)
        all_ce_losses.append(ce_loss)
        all_wce_losses.append(wce_loss)
        all_focal_losses.append(focal_loss)

    avg_ce_loss = np.mean(all_ce_losses)
    avg_wce_loss = np.mean(all_wce_losses)
    avg_focal_loss = np.mean(all_focal_losses)

    print(f'\n{name} - 10-Fold Cross Validation Results')
    print(f'Average Cross-Entropy Loss: {avg_ce_loss:.4f}')
    print(f'Average Weighted Cross-Entropy Loss: {avg_wce_loss:.4f}')
    print(f'Average Focal Loss: {avg_focal_loss:.4f}')
    print(f"\nClassification Report:")
    print(classification_report(all_val_targets, all_val_predictions, digits=4))
