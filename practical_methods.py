import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # LR通常需要标准化
from sklearn.model_selection import KFold
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


def load_and_merge_data(train_path='train.csv'):
    """加载并合并训练集和测试集"""
    train = pd.read_csv(train_path)
    return train


def preprocess_employment_length(data):
    """处理employmentLength字段"""
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(lambda s: np.int8(s.split()[0]) if pd.notnull(s) else s)
    return data


def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])


def process_credit_line(data):
    """处理earliesCreditLine字段"""
    def convert(x):
        try:
            return int(x[-4:])
        except:
            if 'v-' in x:
                return 1900 + int(x.split('-')[1])
            return None
    
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(convert)
    return data


def process_categorical_features(data):
    """处理类别特征"""
    # 需要进行one-hot编码的特征
    onehot_features = ['grade', 'subGrade', 'homeOwnership', 
                      'verificationStatus', 'purpose', 'regionCode']
    
    # 需要进行计数编码的特征
    count_features = ['employmentTitle', 'postCode', 'title']
    
    # 进行one-hot编码
    data = pd.get_dummies(data, columns=onehot_features, drop_first=True)
    
    # 进行计数编码
    for f in count_features:
        data[f + '_cnts'] = data.groupby([f])['id'].transform('count')
        data[f + '_rank'] = data.groupby([f])['id'].rank(ascending=False).fillna(0).astype(int)
        del data[f]
    return data


def prepare_dataset(data):
    """准备训练集和测试集"""
    features = [f for f in data.columns if f not in ['id', 'issueDate', 'isDefault']]
    x_train = data[features]
    y_train = data['isDefault']
    
    return x_train, y_train


def preprocess_pipeline(train_path='train.csv'):
    """完整的预处理流程"""
    # 加载数据
    data = load_and_merge_data(train_path)
    
    # 特征处理
    data = preprocess_employment_length(data)
    data = process_credit_line(data)
    data = process_categorical_features(data)
    
    # 准备数据集
    x_train, y_train = prepare_dataset(data)
    
    return x_train, y_train


# 定义模型训练函数
def cv_model(clf, train_x, train_y, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
        train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs': 24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix],
                              callbacks=[
                                  lgb.early_stopping(stopping_rounds=200),
                                  lgb.log_evaluation(period=200)]
                              )
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)

            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)
            ########
            ########

            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      }

            watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                              early_stopping_rounds=200)
            val_pred = model.predict(valid_matrix, iteration_range=(0, model.best_iteration + 1))

        train[valid_index] = val_pred
        cv_scores.append(roc_auc_score(val_y, val_pred))

        print(cv_scores)

    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train


# 定义LightGBM和XGBoost模型的快捷函数
def lgb_model(x_train, y_train):
    lgb_train = cv_model(lgb, x_train, y_train, "lgb")
    return lgb_train


def xgb_model(x_train, y_train):
    xgb_train = cv_model(xgb, x_train, y_train, "xgb")
    return xgb_train

#############################################################################################################################################

def rf_model(clf, train_x, train_y, clf_name):
    # 添加数据预处理
    # imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
    imputer = SimpleImputer(strategy='most_frequent')  # 使用最频繁值填充

    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    
    train = np.zeros(train_x.shape[0])
    
    cv_scores = []
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        # 对训练集和验证集进行缺失值填充
        trn_x = pd.DataFrame(imputer.fit_transform(trn_x), columns=trn_x.columns)
        val_x = pd.DataFrame(imputer.transform(val_x), columns=val_x.columns)

        if clf_name == "rf":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(trn_x, trn_y)
            val_pred = model.predict_proba(val_x)[:, 1]
            
        train[valid_index] = val_pred
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
    
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train


# 定义随机森林模型的快捷函数
def rf_classifier(x_train, y_train):
    rf_train = rf_model(None, x_train, y_train, "rf")
    return rf_train
############################################################################################################################################

def lr_model(clf, train_x, train_y, clf_name):
    # 添加数据预处理
    imputer = SimpleImputer(strategy='most_frequent')  # 处理缺失值
    scaler = StandardScaler()  # 添加标准化
    
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    
    train = np.zeros(train_x.shape[0])
    
    cv_scores = []
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        # 对训练集和验证集进行缺失值填充
        trn_x = pd.DataFrame(imputer.fit_transform(trn_x), columns=trn_x.columns)
        val_x = pd.DataFrame(imputer.transform(val_x), columns=val_x.columns)
        # 对测试集进行缺失值填充
        
        # 标准化
        trn_x = pd.DataFrame(scaler.fit_transform(trn_x), columns=trn_x.columns)
        val_x = pd.DataFrame(scaler.transform(val_x), columns=val_x.columns)

        if clf_name == "lr":
            model = LogisticRegression(
                C=1.0,  # 正则化强度的倒数
                max_iter=1000,  # 最大迭代次数
                class_weight='balanced',  # 处理类别不平衡
                random_state=seed,
                n_jobs=-1
            )
            model.fit(trn_x, trn_y)
            val_pred = model.predict_proba(val_x)[:, 1]
            
            # 输出特征重要性（可选）
            feature_importance = pd.DataFrame({
                'feature': trn_x.columns,
                'importance': np.abs(model.coef_[0])
            })
            print("\nTop 10 important features:")
            print(feature_importance.sort_values('importance', ascending=False).head(10))
            
        train[valid_index] = val_pred
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
    
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train

# 定义逻辑回归模型的快捷函数
def lr_classifier(x_train, y_train):
    lr_train = lr_model(None, x_train, y_train, "lr")
    return lr_train


class WOETransformer:
    def __init__(self, feature_name, max_bins=10, min_samples=0.05):
        self.feature_name = feature_name
        self.max_bins = max_bins
        self.min_samples = min_samples
        self.bins = None
        self.woe_dict = {}
        self.iv = None
        
    def _calculate_woe_iv(self, x, y):
        # 计算每个分箱的WOE和IV值
        total_positive = np.sum(y == 1)
        total_negative = np.sum(y == 0)
        
        grouped = pd.DataFrame({
            'x': x,
            'y': y
        }).groupby('x')
        
        woe_dict = {}
        iv = 0
        
        for group_value, group_data in grouped:
            positive = np.sum(group_data['y'] == 1)
            negative = np.sum(group_data['y'] == 0)
            
            # 添加平滑处理，避免除零错误
            positive = positive + 0.5
            negative = negative + 0.5
            
            positive_rate = positive / total_positive
            negative_rate = negative / total_negative
            
            woe = np.log(positive_rate / negative_rate)
            iv += (positive_rate - negative_rate) * woe
            
            woe_dict[group_value] = woe
            
        return woe_dict, iv
    
    def fit(self, X, y):
        # 对连续特征进行分箱
        if np.issubdtype(X.dtype, np.number):
            # 使用分位数进行分箱
            quantiles = np.linspace(0, 1, self.max_bins + 1)
            self.bins = np.unique(np.quantile(X, quantiles))
            
            # 确保最小和最大值被包含
            self.bins[0] = float('-inf')
            self.bins[-1] = float('inf')
            
            # 将数据分箱
            binned_x = pd.cut(X, bins=self.bins, labels=range(len(self.bins)-1))
        else:
            # 对于类别特征，直接使用原始类别
            binned_x = X
        
        # 计算WOE值和IV值
        self.woe_dict, self.iv = self._calculate_woe_iv(binned_x, y)
        return self
    
    def transform(self, X):
        if self.bins is not None:
            # 对连续特征进行分箱
            binned_x = pd.cut(X, bins=self.bins, labels=range(len(self.bins)-1))
        else:
            # 对于类别特征，直接使用原始类别
            binned_x = X
        
        # 转换为WOE值
        return binned_x.map(self.woe_dict)

def lr_model_with_woe(clf, train_x, train_y, clf_name):
    # 数据预处理
    imputer = SimpleImputer(strategy='most_frequent')
    
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    
    train = np.zeros(train_x.shape[0])
    
    cv_scores = []
    feature_importance_list = []
    
    # 选择需要进行WOE转换的特征
    numeric_features = train_x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train_x.select_dtypes(include=['object']).columns
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f'************************************ {i + 1} ************************************')
        trn_x, trn_y = train_x.iloc[train_index], train_y[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y[valid_index]
        
        # WOE转换
        woe_transformers = {}
        trn_woe = pd.DataFrame()
        val_woe = pd.DataFrame()

        # 诊断信息：打印原始数据的形状
        print(f"Original training data shape: {trn_x.shape}")
        
        # 对数值特征进行WOE转换
        for feature in numeric_features:
            transformer = WOETransformer(feature)
            transformer.fit(trn_x[feature], trn_y)
            
            # 确保WOE转换后的值为数值类型
            trn_woe[feature] = pd.to_numeric(transformer.transform(trn_x[feature]), errors='coerce')
            val_woe[feature] = pd.to_numeric(transformer.transform(val_x[feature]), errors='coerce')

            # 诊断信息：检查每个特征的NaN数量
            print(f"Feature {feature} NaN count in training set: {trn_woe[feature].isna().sum()}")

            woe_transformers[feature] = transformer
            print(f"Feature {feature} IV: {transformer.iv:.4f}")
        
        # 对类别特征进行WOE转换
        for feature in categorical_features:
            transformer = WOETransformer(feature)
            transformer.fit(trn_x[feature], trn_y)
            
            # 确保WOE转换后的值为数值类型
            trn_woe[feature] = pd.to_numeric(transformer.transform(trn_x[feature]), errors='coerce')
            val_woe[feature] = pd.to_numeric(transformer.transform(val_x[feature]), errors='coerce')

            # 诊断信息：检查每个特征的NaN数量
            print(f"Feature {feature} NaN count in training set: {trn_woe[feature].isna().sum()}")
            
            woe_transformers[feature] = transformer
            print(f"Feature {feature} IV: {transformer.iv:.4f}")

        # 在删除NaN之前打印每列的NaN数量
        print("\nNaN counts before dropna:")
        print(trn_woe.isna().sum())
        
        # 尝试逐列处理NaN: 对于部分NaN的列，使用均值填充;对于全是NaN的列，直接删除该特征
        for col in trn_woe.columns:
            # 使用该列的均值填充NaN
            col_mean = trn_woe[col].mean()
            if pd.isna(col_mean):  # 如果整列都是NaN
                # 删除这个特征
                print(f"Dropping feature {col} due to all NaN values")
                trn_woe = trn_woe.drop(columns=[col])
                val_woe = val_woe.drop(columns=[col])
            else:
                trn_woe[col] = trn_woe[col].fillna(col_mean)
                val_woe[col] = val_woe[col].fillna(col_mean)
        
        # 检查处理后的数据
        print("\nShape after NaN handling:")
        print(f"Training set shape: {trn_woe.shape}")
        print(f"Validation set shape: {val_woe.shape}")
        
        if clf_name == "lr":
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )
            # 使用处理后的数据进行训练
            model.fit(trn_woe, trn_y)
            val_pred = model.predict_proba(val_woe)[:, 1]
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': trn_woe.columns,
                'importance': np.abs(model.coef_[0])
            })
            feature_importance_list.append(feature_importance)
            
            print("\nTop 10 important features:")
            print(feature_importance.sort_values('importance', ascending=False).head(10))
        
        # 存储预测结果
        train[valid_index] = val_pred
        
        cv_scores.append(roc_auc_score(val_y, val_pred))
        print(cv_scores)
    
    # 输出平均特征重要性
    mean_importance = pd.concat(feature_importance_list).groupby('feature').mean()
    print("\nMean feature importance:")
    print(mean_importance.sort_values('importance', ascending=False).head(10))
    
    print(f"{clf_name}_score_list:", cv_scores)
    print(f"{clf_name}_score_mean:", np.mean(cv_scores))
    print(f"{clf_name}_score_std:", np.std(cv_scores))
    return train

# 定义WOE-LR模型的快捷函数
def lr_classifier_with_woe(x_train, y_train):
    lr_train = lr_model_with_woe(None, x_train, y_train, "lr")
    return lr_train


def cnn_model(clf, train_x, train_y, clf_name):
    # 数据预处理
    imputer = SimpleImputer(strategy='most_frequent')
    scaler = StandardScaler()
    
    class TabularDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    class TabularCNN(nn.Module):
        def __init__(self, input_dim):
            super(TabularCNN, self).__init__()
            # 重塑输入维度为 (batch_size, 1, feature_size, 1)
            self.reshape = lambda x: x.view(x.size(0), 1, -1, 1)
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,1), padding=(1,0)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(0.2),
                
                nn.Conv2d(32, 64, kernel_size=(3,1), padding=(1,0)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Dropout(0.2)
            )
            
            # 计算展平后的特征维度
            self.flat_features = 64 * input_dim
            
            self.fc_layers = nn.Sequential(
                nn.Linear(self.flat_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            x = self.reshape(x)
            x = self.conv_layers(x)
            x = x.view(-1, self.flat_features)
            x = self.fc_layers(x)
            return x.squeeze()
    
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    
    train = np.zeros(train_x.shape[0])
    cv_scores = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f'************************************ {i + 1} ************************************')
        trn_x, trn_y = train_x.iloc[train_index], train_y[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y[valid_index]
        
        # 数据预处理
        trn_x = pd.DataFrame(imputer.fit_transform(trn_x), columns=trn_x.columns)
        val_x = pd.DataFrame(imputer.transform(val_x), columns=val_x.columns)
        
        trn_x = pd.DataFrame(scaler.fit_transform(trn_x), columns=trn_x.columns)
        val_x = pd.DataFrame(scaler.transform(val_x), columns=val_x.columns)
        
        # 创建数据加载器
        train_dataset = TabularDataset(trn_x.values, trn_y.values)
        valid_dataset = TabularDataset(val_x.values, val_y.values)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
        
        # 初始化模型
        model = TabularCNN(input_dim=trn_x.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        n_epochs = 10
        best_val_auc = 0
        
        for epoch in range(n_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    val_preds.extend(outputs.cpu().numpy())
                    val_true.extend(batch_y.numpy())
            
            val_auc = roc_auc_score(val_true, val_preds)
            if (epoch + 1) % 2 == 0 or (epoch + 1) == n_epochs:  # 每10轮或最后一轮输出
                print(f'Epoch {epoch+1}/{n_epochs}, Validation AUC: {val_auc:.4f}')
        
        # 最终预测
        model.eval()
        val_predictions = []
        with torch.no_grad():
            for batch_x, _ in valid_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                val_predictions.extend(outputs.cpu().numpy())
        
        train[valid_index] = val_predictions
        cv_score = roc_auc_score(val_y, val_predictions)
        cv_scores.append(cv_score)
        print(f'Fold {i+1} AUC: {cv_score:.4f}')
    
    print(f"{clf_name}_score_list:", cv_scores)
    print(f"{clf_name}_score_mean:", np.mean(cv_scores))
    print(f"{clf_name}_score_std:", np.std(cv_scores))
    return train

# 定义CNN模型的快捷函数
def cnn_classifier(x_train, y_train):
    cnn_train = cnn_model(None, x_train, y_train, "cnn")
    return cnn_train


def main():
    """主函数"""
    # 数据预处理
    x_train, y_train = preprocess_pipeline()
    
    # 模型训练和预测
    print("Training model...")
    # train = lr_classifier_with_woe(x_train, y_train)
    # train = lr_classifier(x_train, y_train)
    train = lgb_model(x_train, y_train)
    # train = xgb_model(x_train, y_train)
    # train = cnn_classifier(x_train, y_train)
    
    # 保存预测结果
    print("Saving predictions...")
    print("Done!")


if __name__ == "__main__":
    main()