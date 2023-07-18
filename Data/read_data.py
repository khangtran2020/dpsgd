import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from Data.datasets import Data


def read_adult(args):
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
              'income']
    label_dict = {
        ' <=50K': '<=50K',
        ' >50K': '>50K',
        ' <=50K.': '<=50K',
        ' >50K.': '>50K'
    }
    train_df = pd.read_csv('Data/Adult/adult.data', header=None)
    test_df = pd.read_csv('Data/Adult/adult.test', skiprows=1, header=None)
    all_data = pd.concat([train_df, test_df], axis=0)
    all_data.columns = header

    def hour_per_week(x):
        if x <= 19:
            return '0'
        elif (x > 19) & (x <= 29):
            return '1'
        elif (x > 29) & (x <= 39):
            return '2'
        elif x > 39:
            return '3'

    def age(x):
        if x <= 24:
            return '0'
        elif (x > 24) & (x <= 34):
            return '1'
        elif (x > 34) & (x <= 44):
            return '2'
        elif (x > 44) & (x <= 54):
            return '3'
        elif (x > 54) & (x <= 64):
            return '4'
        else:
            return '5'

    def country(x):
        if x == ' United-States':
            return 0
        else:
            return 1

    all_data['hours-per-week'] = all_data['hours-per-week'].map(lambda x: hour_per_week(x))
    all_data['age'] = all_data['age'].map(lambda x: age(x))
    all_data['native-country'] = all_data['native-country'].map(lambda x: country(x))
    all_data = all_data.drop(
        ['fnlwgt', 'education-num', 'marital-status', 'occupation', 'relationship', 'capital-gain', 'capital-loss'],
        axis=1)
    temp = pd.get_dummies(all_data['age'], prefix='age')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('age', axis=1)
    temp = pd.get_dummies(all_data['workclass'], prefix='workclass')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('workclass', axis=1)
    temp = pd.get_dummies(all_data['education'], prefix='education')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('education', axis=1)
    temp = pd.get_dummies(all_data['race'], prefix='race')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('race', axis=1)
    temp = pd.get_dummies(all_data['hours-per-week'], prefix='hour')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('hours-per-week', axis=1)
    all_data['income'] = all_data['income'].map(label_dict)
    lb = LabelEncoder()
    all_data['sex'] = lb.fit_transform(all_data['sex'].values)
    lb = LabelEncoder()
    all_data['income'] = lb.fit_transform(all_data['income'].values)
    feature_cols = list(all_data.columns)
    feature_cols.remove('income')
    feature_cols.remove('sex')
    label = 'income'
    if args.mode == 'func':
        all_data = minmax_scale(df=all_data, cols=feature_cols)
        all_data['bias'] = 1.0
        feature_cols.append('bias')
    train_df = all_data[:train_df.shape[0]].reset_index(drop=True)
    test_df = all_data[train_df.shape[0]:].reset_index(drop=True)
    fold_separation(train_df, args.folds, feature_cols, label)
    return train_df, test_df, feature_cols, label


def fold_separation(train_df, folds, feat_cols, label):
    skf = StratifiedKFold(n_splits=folds)
    train_df['fold'] = np.zeros(train_df.shape[0])
    for i, (idxT, idxV) in enumerate(skf.split(train_df[feat_cols], train_df[label])):
        train_df.at[idxV, 'fold'] = i


def minmax_scale(df, cols):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for col in cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df


def init_data(args, fold, train, test):
    df_train = train[train.fold != fold]
    df_valid = train[train.fold == fold]

    # get numpy
    x_tr = df_train[args.feature].values
    y_tr = df_train[args.target].values
    z_tr = df_train[args.z].values

    x_va = df_valid[args.feature].values
    y_va = df_valid[args.target].values
    z_va = df_valid[args.z].values

    x_te = test[args.feature].values
    y_te = test[args.target].values
    z_te = test[args.z].values
    # Defining DataSet

    ## train
    train_dataset = Data(X=x_tr, y=y_tr, ismale=z_tr)

    ## valid
    valid_dataset = Data(X=x_va, y=y_va, ismale=z_va)

    ## test
    test_dataset = Data(X=x_te, y=y_te, ismale=z_te)

    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    tr_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.sampling_rate * len(train_dataset)),
                                            pin_memory=True, drop_last=True, num_workers=0, sampler=sampler)

    va_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                            pin_memory=True, drop_last=False)

    te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                            pin_memory=True, drop_last=False)

    args.n_batch = len(tr_loader)
    args.bs = int(args.sampling_rate * len(train_dataset))
    return tr_loader, va_loader, te_loader
