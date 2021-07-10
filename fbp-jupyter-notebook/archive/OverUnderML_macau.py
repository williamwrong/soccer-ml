#!/usr/bin/env python
# coding: utf-8

from time import strftime

ou_hdc = 'OU25'

import cx_Oracle
import codecs
import os

curr_path = os.getcwd()
db_user = 'JW'
db_password = '901203'
db_dsn = 'HOME-PC/XE'
db_encoding = 'UTF-8'

output_file = '%s\\data\\str_hilo_tune_macau.csv' % (curr_path)
error_file = '%s\\error.txt' % (curr_path)

if os.path.exists(output_file):
    os.remove(output_file)
if os.path.exists(error_file):
    os.remove(error_file)

# Database connection
connection = None
try:
    connection = cx_Oracle.connect(
        db_user,
        db_password,
        db_dsn,
        encoding=db_encoding)

    c = connection.cursor()
    file = codecs.open(output_file, "a+", "utf-8")

    # write file header
    file.write('MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,HOME_FT_GOAL,AWAY_FT_GOAL,ML_TYPE,STR_MODE_HDC,STR_AVG_HI,STR_AVG_LO,STR_MACAU_HI,STR_MACAU_LO,STR_HKJC_HDC,STR_HKJC_HI,STR_HKJC_LO,STR_O_MACAU_HI_DIFF,STR_O_MACAU_LO_DIFF,STR_O_HKJC_HI_DIFF,STR_O_HKJC_LO_DIFF,END_MODE_HDC,END_AVG_HI,END_AVG_LO,END_MACAU_HI,END_MACAU_LO,END_HKJC_HDC,END_HKJC_HI,END_HKJC_LO,END_O_MACAU_HI_DIFF,END_O_MACAU_LO_DIFF,END_O_HKJC_HI_DIFF,END_O_HKJC_LO_DIFF,STR_ASIAN_HDC,STR_A_AVG_HOME,STR_A_AVG_AWAY,STR_A_MACAU_HOME,STR_A_MACAU_AWAY,END_ASIAN_HDC,END_A_AVG_HOME,END_A_AVG_AWAY,END_A_MACAU_HOME,END_A_MACAU_AWAY,HOME_ADV,AWAY_ADV,GAME_POINT,TOTAL_GOAL_COUNT\n')
    sql = """
        SELECT 
            a.MATCH_ID, a.MATCH_DATETIME, a.LEAGUE, a.HOME_TEAM, a.AWAY_TEAM, a.HOME_FT_GOAL, a.AWAY_FT_GOAL, 
            CASE 
                WHEN a.MATCH_DATETIME < TO_TIMESTAMP('2020-06-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'TRAIN'
                WHEN a.MATCH_DATETIME >= TO_TIMESTAMP('2020-06-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') AND a.MATCH_DATETIME < TO_TIMESTAMP('2020-09-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'VAL'
                WHEN a.MATCH_DATETIME >= TO_TIMESTAMP('2020-09-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'TEST'
            END AS ML_TYPE, 
            b.STR_MODE_HDC, ROUND(b.STR_O_AVG_HI,4) AS STR_AVG_HI, ROUND(b.STR_O_AVG_LO,4) AS STR_AVG_LO, 
            b.STR_O_MACAU_HI AS STR_MACAU_HI, b.STR_O_MACAU_LO AS STR_MACAU_LO, 
            b.STR_HKJC_HDC, b.STR_O_HKJC_HI AS STR_HKJC_HI, b.STR_O_HKJC_LO AS STR_HKJC_LO, 
            ROUND((b.STR_O_MACAU_HI-b.STR_O_AVG_HI)/b.STR_O_AVG_HI,4) AS STR_O_MACAU_HI_DIFF, ROUND((b.STR_O_MACAU_LO-b.STR_O_AVG_LO)/b.STR_O_AVG_LO,4) AS STR_O_MACAU_LO_DIFF, 
            ROUND((b.STR_O_HKJC_HI-b.STR_O_AVG_HI)/b.STR_O_AVG_HI,4) AS STR_O_HKJC_HI_DIFF, ROUND((b.STR_O_HKJC_LO-b.STR_O_AVG_LO)/b.STR_O_AVG_LO,4) AS STR_O_HKJC_LO_DIFF, 
            b.END_MODE_HDC, ROUND(b.END_O_AVG_HI,4) AS END_AVG_HI, ROUND(b.END_O_AVG_LO,4) AS END_AVG_LO, 
            b.END_O_MACAU_HI AS END_MACAU_HI, b.END_O_MACAU_LO AS END_MACAU_LO, 
            b.END_HKJC_HDC, b.END_O_HKJC_HI AS END_HKJC_HI, b.END_O_HKJC_LO AS END_HKJC_LO, 
            ROUND((b.END_O_MACAU_HI-b.END_O_AVG_HI)/b.END_O_AVG_HI,4) AS END_O_MACAU_HI_DIFF, ROUND((b.END_O_MACAU_LO-b.END_O_AVG_LO)/b.END_O_AVG_LO,4) AS END_O_MACAU_LO_DIFF, 
            ROUND((b.END_O_HKJC_HI-b.END_O_AVG_HI)/b.END_O_AVG_HI,4) AS END_O_HKJC_HI_DIFF, ROUND((b.END_O_HKJC_LO-b.END_O_AVG_LO)/b.END_O_AVG_LO,4) AS END_O_HKJC_LO_DIFF, 
            c.STR_MODE_HDC AS STR_ASIAN_HDC, ROUND(c.STR_O_AVG_HOME,4) AS STR_A_AVG_HOME, ROUND(c.STR_O_AVG_AWAY,4) AS STR_A_AVG_AWAY, c.STR_O_MACAU_H AS STR_A_MACAU_HOME, c.STR_O_MACAU_A AS STR_A_MACAU_AWAY, 
            c.END_MODE_HDC AS END_ASIAN_HDC, ROUND(c.END_O_AVG_HOME,4) AS END_A_AVG_HOME, ROUND(c.END_O_AVG_AWAY,4) AS END_A_AVG_AWAY, c.END_O_MACAU_H AS END_A_MACAU_HOME, c.END_O_MACAU_A AS END_A_MACAU_AWAY, 
            (d.HOME_HOME_GF+d.AWAY_AWAY_GA)/10 AS HOME_ADV, (d.HOME_HOME_GA+d.AWAY_AWAY_GF)/10 AS AWAY_ADV, (d.HOME_HOME_GF+d.AWAY_AWAY_GA)/10+(d.HOME_HOME_GA+d.AWAY_AWAY_GF)/10 AS GAME_POINT, 
            a.HOME_FT_GOAL+a.AWAY_FT_GOAL AS TOTAL_GOAL_COUNT
        FROM MATCH_INFO a, HILO_MERGE2 b, ASIAN_MERGE c, RECENT_RAW d
        WHERE a.MATCH_ID=b.MATCH_ID AND a.MATCH_ID=c.MATCH_ID AND a.MATCH_ID=d.MATCH_ID
            AND b.STR_MODE_HDC=b.STR_MACAU_HDC
            AND c.STR_MODE_HDC=c.STR_MACAU_HDC 
            -- AND a.HOME_FT_GOAL IS NOT NULL
        ORDER BY a.MATCH_DATETIME
        """
        
    c.execute(sql)
    result = c.fetchall()
    for row in result:
        ft_home_goal = row[5]
        ft_away_goal = row[6]
        total_goal = row[45]
        if row[5] == None:
            ft_home_goal = ''
        if row[6] == None:
            ft_away_goal = ''
        if row[45] == None:
            total_goal = ''
        file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (row[0],row[1],row[2],row[3],row[4],ft_home_goal,ft_away_goal,row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21],row[22],row[23],row[24],row[25],row[26],row[27],row[28],row[29],row[30],row[31],row[32],row[33],row[34],row[35],row[36],row[37],row[38],row[39],row[40],row[41],row[42],row[43],row[44],total_goal))

    file.close()

except cx_Oracle.Error as error:
    file.close()
    err_file = codecs.open(error_file, 'a+', 'utf-8')
    err_file.write('[%s] Oracle error - %s\n' % (strftime('%Y-%m-%d %H:%M:%S'), error))
    err_file.close()
except:
    file.close()


import pandas as pd
from datetime import datetime


pd.set_option('display.width', 3000)
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 3000)


#load datasets
df = pd.read_csv('data/str_hilo_tune_macau.csv', sep=',')


# #final feature selection
df = df[['MATCH_ID','MATCH_DATETIME','LEAGUE','HOME_TEAM','AWAY_TEAM','HOME_FT_GOAL','AWAY_FT_GOAL','ML_TYPE','STR_MODE_HDC','STR_AVG_HI','STR_AVG_LO','STR_MACAU_HI','STR_MACAU_LO','STR_HKJC_HDC','STR_HKJC_HI','STR_HKJC_LO','STR_O_MACAU_HI_DIFF','STR_O_MACAU_LO_DIFF','STR_O_HKJC_HI_DIFF','STR_O_HKJC_LO_DIFF','END_MODE_HDC','END_AVG_HI','END_AVG_LO','END_MACAU_HI','END_MACAU_LO','END_HKJC_HDC','END_HKJC_HI','END_HKJC_LO','END_O_MACAU_HI_DIFF','END_O_MACAU_LO_DIFF','END_O_HKJC_HI_DIFF','END_O_HKJC_LO_DIFF','STR_ASIAN_HDC','STR_A_AVG_HOME','STR_A_AVG_AWAY','STR_A_MACAU_HOME','STR_A_MACAU_AWAY','END_ASIAN_HDC','END_A_AVG_HOME','END_A_AVG_AWAY','END_A_MACAU_HOME','END_A_MACAU_AWAY','HOME_ADV','AWAY_ADV','GAME_POINT','TOTAL_GOAL_COUNT']]
df['OU25']= [1 if x > 2.5 else 0 for x in df['TOTAL_GOAL_COUNT']]
df1=df.drop(['MATCH_ID','MATCH_DATETIME','LEAGUE','HOME_TEAM','AWAY_TEAM','HOME_FT_GOAL','AWAY_FT_GOAL','TOTAL_GOAL_COUNT'], axis=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import category_encoders as ce
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#Split train and test datasets
X = df1
X_train = df1.query('ML_TYPE == "TRAIN"')
X_val = df1.query('ML_TYPE == "VAL"')
X_test = df1.query('ML_TYPE == "TEST"')
y_train = X_train.pop(ou_hdc)
y_val = X_val.pop(ou_hdc)
y_test = X_test.pop(ou_hdc)


# Baseline
y_train.value_counts(normalize=True)


transformers = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='median')
)

X_train_transformed = transformers.fit_transform(X_train)
X_val_transformed = transformers.transform(X_val)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_transformed, y_train)


import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(
    model,
    scoring='accuracy',
    n_iter=5,
    random_state=42
)

permuter.fit(X_val_transformed, y_val)


eli5.show_weights(
    permuter,
    top=None,
    feature_names=X_test.columns.tolist()
)


from xgboost import XGBClassifier
model1 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    XGBClassifier(n_estimators=20, random_state=42, n_jobs=-1)
)

model1.fit(X_train, y_train)


print('XGBClassifier - Training Accuracy:', model1.score(X_train, y_train))
print('XGBClassifier - Validation Accuracy:', model1.score(X_val, y_val))
print('XGBClassifier - Test Accuracy:', model1.score(X_test, y_test))



model2 = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='median'), 
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)
model2.fit(X_train, y_train)

print('RandomForestClassifier - Training accuracy:', model2.score(X_train, y_train))
print('RandomForestClassifier - Validation accuracy:', model2.score(X_val, y_val))
print('RandomForestClassifier - Test accuracy:', model2.score(X_test, y_test))


model3 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    LogisticRegression(max_iter=300, random_state=42)
)

model3.fit(X_train, y_train)

print('LogisticRegression - Training Accuracy:', model3.score(X_train, y_train))
print('LogisticRegression - Validation Accuracy:', model3.score(X_val, y_val))
print('LogisticRegression - Test Accuracy:', model3.score(X_test, y_test))


#remove negative features
X_train = X_train[['END_ASIAN_HDC','END_O_HKJC_HI_DIFF','HOME_ADV','STR_A_AVG_AWAY','STR_AVG_HI','END_O_MACAU_HI_DIFF','END_HKJC_HI','STR_AVG_LO','END_A_AVG_HOME','END_O_HKJC_LO_DIFF']]
X_val = X_val[['END_ASIAN_HDC','END_O_HKJC_HI_DIFF','HOME_ADV','STR_A_AVG_AWAY','STR_AVG_HI','END_O_MACAU_HI_DIFF','END_HKJC_HI','STR_AVG_LO','END_A_AVG_HOME','END_O_HKJC_LO_DIFF']]
X_test = X_test[['END_ASIAN_HDC','END_O_HKJC_HI_DIFF','HOME_ADV','STR_A_AVG_AWAY','STR_AVG_HI','END_O_MACAU_HI_DIFF','END_HKJC_HI','STR_AVG_LO','END_A_AVG_HOME','END_O_HKJC_LO_DIFF']]


from xgboost import XGBClassifier
model1 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    XGBClassifier(n_estimators=12, random_state=42, n_jobs=-1, learning_rate=0.097, subsample=1)
)

model1.fit(X_train, y_train)

print('XGBClassifier - Training Accuracy:', model1.score(X_train, y_train))
print('XGBClassifier - Validation Accuracy:', model1.score(X_val, y_val))
print('XGBClassifier - Test Accuracy:', model1.score(X_test, y_test))


model2 = Pipeline([
                  ('oe', ce.OrdinalEncoder()),
                  ('impute', SimpleImputer(strategy='mean')),
                  ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

model2.fit(X_train, y_train)

print('RandomForestClassifier - Training accuracy:', model2.score(X_train, y_train))
print('RandomForestClassifier - Validation accuracy:', model2.score(X_val, y_val))
print('RandomForestClassifier - Test accuracy:', model2.score(X_test, y_test))


transformers = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='median')
)

X_train_transformed = transformers.fit_transform(X_train)
X_val_transformed = transformers.transform(X_val)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_transformed, y_train)


import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(
    model,
    scoring='accuracy',
    n_iter=5,
    random_state=42
)

permuter.fit(X_val_transformed, y_val)


eli5.show_weights(
    permuter,
    top=None,
    feature_names=X_test.columns.tolist()
)


model3 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    LogisticRegression(random_state=42)
)

model3.fit(X_train, y_train)

print('LogisticRegression - Training Accuracy:', model3.score(X_train, y_train))
print('LogisticRegression - Validation Accuracy:', model3.score(X_val, y_val))
print('LogisticRegression - Test Accuracy:', model3.score(X_test, y_test))


final = df[['MATCH_ID','MATCH_DATETIME', 'ML_TYPE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'END_MODE_HDC', 'END_MACAU_HI', 'END_MACAU_LO', 'END_HKJC_HDC', 'END_HKJC_HI', 'END_HKJC_LO', 'TOTAL_GOAL_COUNT', ou_hdc]]
final = final.query('ML_TYPE == "TEST"')

y_pred = model1.predict(X_test)
class_probabilities = model1.predict_proba(X_test)

pred = pd.DataFrame(y_pred, columns=['pred'])
prob = pd.DataFrame(class_probabilities, columns=['prob0','prob1' ])

final.reset_index(drop=True, inplace=True)
pred.reset_index(drop=True, inplace=True)
prob.reset_index(drop=True, inplace=True)

fin = pd.concat([final, prob, pred], axis=1)

hdc_col = 'END_MODE_HDC'
hi_col = 'END_MACAU_HI'
lo_col = 'END_MACAU_LO'
hkjc_hdc = 2.5


# Match date prediction
from datetime import datetime, timedelta, date, time
current_match_date = datetime.now()
current_time = datetime.now().time()
if current_time >= time(0,0) and current_time <= time(11, 30):
    previous_day = current_match_date - timedelta(days=1)
    current_match_date = previous_day
current_match_date = current_match_date.strftime("%Y-%m-%d 11:30:00")

# print('MATCH_DATETIME > \'%s\'' % current_match_date)
fin = fin.query('MATCH_DATETIME > \'%s\'' % current_match_date)

# Google sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# from pprint import pprint

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("overunderml.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ML OverUnder").sheet1
data = sheet.get_all_records()

start_row = 4
end_row = sheet.row_count
if end_row >= start_row:
    sheet.delete_rows(start_row,end_row)

exec_time = strftime('%Y-%m-%d %H:%M:%S')

data_row = []
for i,j in fin.iterrows():
    col_match_id = fin.loc[i, 'MATCH_ID']
    col_match_datetime = fin.loc[i, 'MATCH_DATETIME']
    col_home_team = fin.loc[i, 'HOME_TEAM']
    col_away_team = fin.loc[i, 'AWAY_TEAM']
    col_end_mode_hdc = fin.loc[i, 'END_MODE_HDC']
    col_end_macau_hi = fin.loc[i, 'END_MACAU_HI']
    col_end_macau_lo = fin.loc[i, 'END_MACAU_LO']
    col_end_hkjc_hdc = fin.loc[i, 'END_HKJC_HDC']
    col_end_hkjc_hi = fin.loc[i, 'END_HKJC_HI']
    col_end_hkjc_lo = fin.loc[i, 'END_HKJC_LO']
    col_prob_0 = fin.loc[i, 'prob0']
    col_prob_1 = fin.loc[i, 'prob1']
    col_pred = fin.loc[i, 'pred']
    data_row.append([int(col_match_id),str(col_match_datetime),str(col_home_team),str(col_away_team),float(col_end_mode_hdc),float(col_end_macau_hi),float(col_end_macau_lo),float(col_end_hkjc_hdc),float(col_end_hkjc_hi),float(col_end_hkjc_lo),float(col_prob_0),float(col_prob_1),int(col_pred)])

sheet.append_rows(data_row)
print('Last update time: %s' % exec_time)
sheet.update_cell(1,2,exec_time)
