import numpy as np
import pandas as pd

import datetime
from datetime import datetime

from sklearn import model_selection

import lightgbm as lgb

#
# DATA LOADING
#

train = pd.read_csv('/mnt/churn_prediction/train_v2.csv')
user_logs = pd.read_csv('/mnt/churn_prediction/user_logs_v2.csv')
members = pd.read_csv('/mnt/churn_prediction/members_v3.csv')
transactions = pd.read_csv('/mnt/churn_prediction/transactions_v2.csv')

# print(train.shape)  # (970960, 2)
# print(user_logs.shape)  # (18396362, 9)
# print(members.shape)  # (6769473, 6)
# print(transactions.shape)  # (1431009, 9)

user_logs_sum_data = user_logs.groupby('msno').sum()
user_logs_count_data = pd.DataFrame(user_logs.groupby('msno').date.count().reset_index())
user_logs_count_data.columns = ['msno', 'used_days']
user_logs_new_data = pd.merge(user_logs_sum_data, user_logs_count_data, how = 'inner', left_on = 'msno', right_on = 'msno')

del user_logs_new_data['date']

all_data = pd.merge(train, user_logs_new_data, how='left', left_on='msno', right_on='msno')

del transactions['transaction_date']
del transactions['membership_expire_date']

transactions_mean_data = transactions.groupby('msno').mean()
transactions_mean_data = transactions_mean_data[['payment_plan_days', 'plan_list_price', 'actual_amount_paid']]

all_data = pd.merge(all_data, transactions_mean_data, how='left', left_on='msno', right_on='msno')

gender_new = [1 if x == 'male' else 0 for x in members.gender ]
members['gender_new'] = gender_new

current = datetime.strptime('20170331', "%Y%m%d").date()
members['num_days'] = members.registration_init_time.apply(
    lambda x: (current - datetime.strptime(str(int(x)), "%Y%m%d").date()).days if pd.notnull(x) else "NAN"
)

members_all_data = members

del members_all_data['city']
del members_all_data['bd']
del members_all_data['registered_via']
del members_all_data['registration_init_time']
del members_all_data['gender']

all_data = pd.merge(all_data, members_all_data, how='left', left_on='msno', right_on='msno')

all_data = all_data.fillna(-1)

cols = [c for c in all_data.columns if c not in ['is_churn','msno']]
X = all_data[cols]
Y = all_data['is_churn']

test_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=test_size, random_state=seed
)

validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X_train, Y_train, test_size=test_size, random_state=seed
)

#
# MODEL TRAINING
#

lgb_params = {
    'learning_rate': 0.01,
    'application': 'binary',
    'max_depth': 40,
    'num_leaves': 3300,
    'verbosity': -1,
    'metric': 'binary_logloss'
}
d_train = lgb.Dataset(X_train, label=Y_train)
d_valid = lgb.Dataset(X_validation, label=Y_validation)
watchlist = [d_train, d_valid]

model = lgb.train(
    lgb_params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist,
    early_stopping_rounds=50, verbose_eval=10
)

#
# SAVE MODEL TO DISK.
#
model.save_model('/spell/lgb_classifier.txt', num_iteration=model.best_iteration)
