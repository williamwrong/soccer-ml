#!/usr/bin/env python
# coding: utf-8

import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
import category_encoders as ce
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pandas as pd
import codecs
import cx_Oracle
from datetime import datetime, timedelta, date, time
from time import strftime
import os

curr_path = os.getcwd()
csv_name = 'str_hilo_35.csv'
output_file = '%s\\data\\%s' % (curr_path, csv_name)
error_file = '%s\\error.txt' % (curr_path)

if os.path.exists(error_file):
    os.remove(error_file)

# Match date
current_match_date = datetime.now()
current_time = datetime.now().time()
if current_time >= time(0, 0) and current_time <= time(11, 30):
    previous_day = current_match_date - timedelta(days=1)
    current_match_date = previous_day
db_match_date = current_match_date.strftime("%Y%m%d")
current_match_date = current_match_date.strftime("%Y-%m-%d 11:30:00")

ou_hdc = 'OU35'

print('Execution time: %s' % strftime('%Y-%m-%d %H:%M:%S'))
print('Current match date: %s' % current_match_date)


db_user = 'JW'
db_password = '901203'
db_dsn = 'HOME-PC/XE'
db_encoding = 'UTF-8'

# Database connection
connection = None
try:
    connection = cx_Oracle.connect(
        db_user,
        db_password,
        db_dsn,
        encoding=db_encoding)

    c = connection.cursor()

    if os.path.exists(output_file):
        os.remove(output_file)

    file = codecs.open(output_file, "a+", "utf-8")

    # write file header
    file.write('MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,HOME_FT_GOAL,AWAY_FT_GOAL,TOTAL_GOAL_COUNT,ML_TYPE,STR_OU_MODE_HDC,STR_OU_AVG_HI,STR_OU_AVG_LO,STR_OU_BET365_HDC,STR_OU_BET365_HI,STR_OU_BET365_LO,STR_OU_YINGWO_HDC,STR_OU_YINGWO_HI,STR_OU_YINGWO_LO,STR_OU_PINNACLE_HDC,STR_OU_PINNACLE_HI,STR_OU_PINNACLE_LO,STR_OU_MACAU_HDC,STR_OU_MACAU_HI,STR_OU_MACAU_LO,STR_OU_HKJC_HDC,STR_OU_HKJC_HI,STR_OU_HKJC_LO,STR_O_BET365_HI_DIFF,STR_O_BET365_LO_DIFF,STR_O_YINGWO_HI_DIFF,STR_O_YINGWO_LO_DIFF,STR_O_PINNACLE_HI_DIFF,STR_O_PINNACLE_LO_DIFF,STR_O_MACAU_HI_DIFF,STR_O_MACAU_LO_DIFF,STR_O_HKJC_HI_DIFF,STR_O_HKJC_LO_DIFF,STR_A_MODE_HDC,STR_A_AVG_H,STR_A_AVG_A,STR_A_BET365_HDC,STR_A_BET365_H,STR_A_BET365_A,STR_A_YINGWO_HDC,STR_A_YINGWO_H,STR_A_YINGWO_A,STR_A_PINNACLE_HDC,STR_A_PINNACLE_H,STR_A_PINNACLE_A,STR_A_MACAU_HDC,STR_A_MACAU_H,STR_A_MACAU_A,STR_HDA_BET365_H,STR_HDA_BET365_D,STR_HDA_BET365_A,STR_HDA_YINGWO_H,STR_HDA_YINGWO_D,STR_HDA_YINGWO_A,STR_HDA_PINNACLE_H,STR_HDA_PINNACLE_D,STR_HDA_PINNACLE_A,STR_HDA_MACAU_H,STR_HDA_MACAU_D,STR_HDA_MACAU_A,STR_HDA_HKJC_H,STR_HDA_HKJC_D,STR_HDA_HKJC_A,HOME_TOTAL_GF,HOME_TOTAL_GA,HOME_AVG_GF,HOME_HOME_GF,HOME_HOME_GA,HOME_HOME_AVG_GF,AWAY_TOTAL_GF,AWAY_TOTAL_GA,AWAY_AVG_GF,AWAY_AWAY_GF,AWAY_AWAY_GA,AWAY_AWAY_AVG_GF,HOME_ADV,AWAY_ADV,GAME_POINT,HOME_WIN_RATE,HOME_DRAW_RATE,HOME_LOSE_RATE,HOME_HOME_WIN_RATE,HOME_HOME_LOSE_RATE,HOME_HOME_DRAW_RATE,AWAY_WIN_RATE,AWAY_DRAW_RATE,AWAY_LOSE_RATE,AWAY_AWAY_WIN_RATE,AWAY_AWAY_LOSE_RATE,AWAY_AWAY_DRAW_RATE\n')
    sql = """
        SELECT 
            -- INFO
            info.MATCH_ID, info.MATCH_DATETIME, info.LEAGUE, info.HOME_TEAM, info.AWAY_TEAM, info.HOME_FT_GOAL, info.AWAY_FT_GOAL, info.HOME_FT_GOAL+info.AWAY_FT_GOAL AS TOTAL_GOAL_COUNT,
            CASE 
                WHEN info.MATCH_DATETIME < TO_TIMESTAMP('2020-06-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'TRAIN'
                WHEN info.MATCH_DATETIME >= TO_TIMESTAMP('2020-06-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') AND info.MATCH_DATETIME < TO_TIMESTAMP('2020-09-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'VALID'
                WHEN info.MATCH_DATETIME >= TO_TIMESTAMP('2020-09-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') AND info.MATCH_DATETIME < TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') THEN 'TEST'
                WHEN info.MATCH_DATETIME >= TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') THEN 'PREDICT'
            END AS ML_TYPE, 
            -- HILO
            hilo.STR_MODE_HDC AS STR_OU_MODE_HDC, ROUND(hilo.STR_O_AVG_HI,4) AS STR_OU_AVG_HI, ROUND(hilo.STR_O_AVG_LO,4) AS STR_OU_AVG_LO, 
            hilo.STR_BET365_HDC AS STR_OU_BET365_HDC, hilo.STR_O_BET365_HI AS STR_OU_BET365_HI, hilo.STR_O_BET365_LO AS STR_OU_BET365_LO, 
            hilo.STR_YINGYO_HDC AS STR_OU_YINGWO_HDC, hilo.STR_O_YINGYO_HI AS STR_OU_YINGWO_HI, hilo.STR_O_YINGYO_LO AS STR_OU_YINGWO_LO, 
            hilo.STR_PINGBOK_HDC AS STR_OU_PINNACLE_HDC, hilo.STR_O_PINGBOK_HI AS STR_OU_PINNACLE_HI, hilo.STR_O_PINGBOK_LO AS STR_OU_PINNACLE_LO, 
            hilo.STR_MACAU_HDC AS STR_OU_MACAU_HDC, hilo.STR_O_MACAU_HI AS STR_OU_MACAU_HI, hilo.STR_O_MACAU_LO AS STR_OU_MACAU_LO, 
            hilo.STR_HKJC_HDC AS STR_OU_HKJC_HDC, hilo.STR_O_HKJC_HI AS STR_OU_HKJC_HI, hilo.STR_O_HKJC_LO AS STR_OU_HKJC_LO, 
        --    CASE WHEN hilo.STR_MACAU_HDC=hilo.STR_MODE_HDC AND hilo.STR_O_MACAU_HI<hilo.STR_O_AVG_LO AND hilo.STR_O_MACAU_LO<hilo.STR_O_AVG_HI THEN 1 ELSE 0 END AS STR_OU_MACAU_WRONG_HDC, 
        --    CASE WHEN hilo.STR_HKJC_HDC=hilo.STR_MODE_HDC AND hilo.STR_O_HKJC_HI<hilo.STR_O_AVG_LO AND hilo.STR_O_HKJC_LO<hilo.STR_O_AVG_HI THEN 1 ELSE 0 END AS STR_OU_HKJC_WRONG_HDC, 
            -- HILO DIFF
            ROUND((hilo.STR_O_BET365_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_O_BET365_HI_DIFF, ROUND((hilo.STR_O_BET365_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_O_BET365_LO_DIFF, 
            ROUND((hilo.STR_O_YINGYO_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_O_YINGWO_HI_DIFF, ROUND((hilo.STR_O_YINGYO_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_O_YINGWO_LO_DIFF, 
            ROUND((hilo.STR_O_PINGBOK_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_O_PINNACLE_HI_DIFF, ROUND((hilo.STR_O_PINGBOK_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_O_PINNACLE_LO_DIFF, 
            ROUND((hilo.STR_O_MACAU_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_O_MACAU_HI_DIFF, ROUND((hilo.STR_O_MACAU_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_O_MACAU_LO_DIFF, 
            ROUND((hilo.STR_O_HKJC_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_O_HKJC_HI_DIFF, ROUND((hilo.STR_O_HKJC_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_O_HKJC_LO_DIFF, 
            -- ASIAN
            asian.STR_MODE_HDC AS STR_A_MODE_HDC, ROUND(asian.STR_O_AVG_HOME,4) AS STR_A_AVG_H, ROUND(asian.STR_O_AVG_AWAY,4) AS STR_A_AVG_A, 
            asian.STR_BET365_HDC AS STR_A_BET365_HDC, asian.STR_O_BET365_H AS STR_A_BET365_H, asian.STR_O_BET365_A AS STR_A_BET365_A, 
            asian.STR_YINGYO_HDC AS STR_A_YINGWO_HDC, asian.STR_O_YINGYO_H AS STR_A_YINGWO_H, asian.STR_O_YINGYO_A AS STR_A_YINGWO_A, 
            asian.STR_PINGBOK_HDC AS STR_A_PINNACLE_HDC, asian.STR_O_PINGBOK_H AS STR_A_PINNACLE_H, asian.STR_O_PINGBOK_A AS STR_A_PINNACLE_A, 
            asian.STR_MACAU_HDC AS STR_A_MACAU_HDC, asian.STR_O_MACAU_H AS STR_A_MACAU_H, asian.STR_O_MACAU_A AS STR_A_MACAU_A, 
            -- HDA
            bet365.HOME_ODD AS STR_HDA_BET365_H, bet365.DRAW_ODD AS STR_HDA_BET365_D, bet365.AWAY_ODD AS STR_HDA_BET365_A, 
            yingwo.HOME_ODD AS STR_HDA_YINGWO_H, yingwo.DRAW_ODD AS STR_HDA_YINGWO_D, yingwo.AWAY_ODD AS STR_HDA_YINGWO_A, 
            pinnacle.HOME_ODD AS STR_HDA_PINNACLE_H, pinnacle.DRAW_ODD AS STR_HDA_PINNACLE_D, pinnacle.AWAY_ODD AS STR_HDA_PINNACLE_A, 
            macau.HOME_ODD AS STR_HDA_MACAU_H, macau.DRAW_ODD AS STR_HDA_MACAU_D, macau.AWAY_ODD AS STR_HDA_MACAU_A, 
            hkjc.HOME_ODD AS STR_HDA_HKJC_H, hkjc.DRAW_ODD AS STR_HDA_HKJC_D, hkjc.AWAY_ODD AS STR_HDA_HKJC_A, 
            -- RECENT STAT
            recent.HOME_TOTAL_GF, recent.HOME_TOTAL_GA, recent.HOME_AVG_GF, recent.HOME_HOME_GF, recent.HOME_HOME_GA, recent.HOME_HOME_AVG_GF, 
            recent.AWAY_TOTAL_GF, recent.AWAY_TOTAL_GA, recent.AWAY_AVG_GF, recent.AWAY_AWAY_GF, recent.AWAY_AWAY_GA, recent.AWAY_AWAY_AVG_GF, 
            (recent.HOME_HOME_GF+recent.AWAY_AWAY_GA)/10 AS HOME_ADV, (recent.HOME_HOME_GA+recent.AWAY_AWAY_GF)/10 AS AWAY_ADV, (recent.HOME_HOME_GF+recent.AWAY_AWAY_GA)/10+(recent.HOME_HOME_GA+recent.AWAY_AWAY_GF)/10 AS GAME_POINT,
            recent.HOME_WIN_RATE, recent.HOME_DRAW_RATE, recent.HOME_LOSE_RATE, recent.HOME_HOME_WIN_RATE, recent.HOME_HOME_LOSE_RATE, recent.HOME_HOME_DRAW_RATE, 
            recent.AWAY_WIN_RATE, recent.AWAY_DRAW_RATE, recent.AWAY_LOSE_RATE, recent.AWAY_AWAY_WIN_RATE, recent.AWAY_AWAY_LOSE_RATE, recent.AWAY_AWAY_DRAW_RATE
        FROM 
            MATCH_INFO info, HILO_MERGE2 hilo, ASIAN_MERGE asian, RECENT_RAW recent, HDA_RAW macau, HDA_RAW bet365, HDA_RAW yingwo, HDA_RAW pinnacle, HDA_RAW hkjc
        WHERE 
            info.MATCH_ID=hilo.MATCH_ID AND info.MATCH_ID=asian.MATCH_ID AND info.MATCH_ID=recent.MATCH_ID AND info.MATCH_ID=macau.MATCH_ID AND info.MATCH_ID=bet365.MATCH_ID AND info.MATCH_ID=yingwo.MATCH_ID AND info.MATCH_ID=pinnacle.MATCH_ID AND info.MATCH_ID=hkjc.MATCH_ID 
            AND macau.BOOKMAKER='澳门' AND bet365.BOOKMAKER='bet365' AND yingwo.BOOKMAKER='盈禾' AND pinnacle.BOOKMAKER='Pinnacle' AND hkjc.BOOKMAKER='香港马会' 
            AND macau.HANDICAP_TYPE=0 AND bet365.HANDICAP_TYPE=0 AND yingwo.HANDICAP_TYPE=0 AND pinnacle.HANDICAP_TYPE=0 AND hkjc.HANDICAP_TYPE=0 
            -- AND hilo.STR_MODE_HDC=hilo.STR_MACAU_HDC
            AND hilo.STR_MODE_HDC>=3 
            AND info.LEAGUE NOT IN ('歐國聯','欧青U21外')
            -- AND asian.STR_MODE_HDC=asian.STR_MACAU_HDC 
            -- AND info.HOME_FT_GOAL IS NOT NULL
            AND (info.MATCH_DATETIME >= TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') OR (info.MATCH_DATETIME < TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') AND info.HOME_FT_GOAL IS NOT NULL))
        ORDER BY info.MATCH_DATETIME, info.MATCH_ID
        """ % (current_match_date, current_match_date, current_match_date, current_match_date)

    c.execute(sql)
    result = c.fetchall()
    for row in result:
        ft_home_goal = row[5]
        ft_away_goal = row[6]
        total_goal = row[7]
        if row[5] == None:
            ft_home_goal = ''
        if row[6] == None:
            ft_away_goal = ''
        if row[7] == None:
            total_goal = ''
        file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (row[0], row[1], row[2], row[3], row[4], ft_home_goal, ft_away_goal, total_goal, row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25], row[26], row[27], row[28], row[29], row[
                   30], row[31], row[32], row[33], row[34], row[35], row[36], row[37], row[38], row[39], row[40], row[41], row[42], row[43], row[44], row[45], row[46], row[47], row[48], row[49], row[50], row[51], row[52], row[53], row[54], row[55], row[56], row[57], row[58], row[59], row[60], row[61], row[62], row[63], row[64], row[65], row[66], row[67], row[68], row[69], row[70], row[71], row[72], row[73], row[74], row[75], row[76], row[77], row[78], row[79], row[80], row[81], row[82], row[83], row[84], row[85], row[86], row[87], row[88], row[89], row[90], row[91], row[92], row[93]))

    file.close()

except cx_Oracle.Error as error:
    file.close()
    err_file = codecs.open(error_file, 'a+', 'utf-8')
    err_file.write('[%s] Oracle error - %s\n' %
                   (strftime('%Y-%m-%d %H:%M:%S'), error))
    err_file.close()
except:
    file.close()


pd.set_option('display.width', 3000)
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 3000)


# load datasets
df = pd.read_csv('data/%s' % csv_name, sep=',')


# final feature selection
df = df[['MATCH_ID', 'MATCH_DATETIME', 'LEAGUE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'TOTAL_GOAL_COUNT', 'ML_TYPE', 'STR_OU_MODE_HDC', 'STR_OU_AVG_HI', 'STR_OU_AVG_LO', 'STR_OU_BET365_HDC', 'STR_OU_BET365_HI', 'STR_OU_BET365_LO', 'STR_OU_YINGWO_HDC', 'STR_OU_YINGWO_HI', 'STR_OU_YINGWO_LO', 'STR_OU_PINNACLE_HDC', 'STR_OU_PINNACLE_HI', 'STR_OU_PINNACLE_LO', 'STR_OU_MACAU_HDC', 'STR_OU_MACAU_HI', 'STR_OU_MACAU_LO', 'STR_OU_HKJC_HDC', 'STR_OU_HKJC_HI', 'STR_OU_HKJC_LO', 'STR_O_BET365_HI_DIFF', 'STR_O_BET365_LO_DIFF', 'STR_O_YINGWO_HI_DIFF', 'STR_O_YINGWO_LO_DIFF', 'STR_O_PINNACLE_HI_DIFF', 'STR_O_PINNACLE_LO_DIFF', 'STR_O_MACAU_HI_DIFF', 'STR_O_MACAU_LO_DIFF', 'STR_O_HKJC_HI_DIFF', 'STR_O_HKJC_LO_DIFF', 'STR_A_MODE_HDC', 'STR_A_AVG_H', 'STR_A_AVG_A', 'STR_A_BET365_HDC', 'STR_A_BET365_H', 'STR_A_BET365_A', 'STR_A_YINGWO_HDC', 'STR_A_YINGWO_H', 'STR_A_YINGWO_A',
         'STR_A_PINNACLE_HDC', 'STR_A_PINNACLE_H', 'STR_A_PINNACLE_A', 'STR_A_MACAU_HDC', 'STR_A_MACAU_H', 'STR_A_MACAU_A', 'STR_HDA_BET365_H', 'STR_HDA_BET365_D', 'STR_HDA_BET365_A', 'STR_HDA_YINGWO_H', 'STR_HDA_YINGWO_D', 'STR_HDA_YINGWO_A', 'STR_HDA_PINNACLE_H', 'STR_HDA_PINNACLE_D', 'STR_HDA_PINNACLE_A', 'STR_HDA_MACAU_H', 'STR_HDA_MACAU_D', 'STR_HDA_MACAU_A', 'STR_HDA_HKJC_H', 'STR_HDA_HKJC_D', 'STR_HDA_HKJC_A', 'HOME_TOTAL_GF', 'HOME_TOTAL_GA', 'HOME_AVG_GF', 'HOME_HOME_GF', 'HOME_HOME_GA', 'HOME_HOME_AVG_GF', 'AWAY_TOTAL_GF', 'AWAY_TOTAL_GA', 'AWAY_AVG_GF', 'AWAY_AWAY_GF', 'AWAY_AWAY_GA', 'AWAY_AWAY_AVG_GF', 'HOME_ADV', 'AWAY_ADV', 'GAME_POINT', 'HOME_WIN_RATE', 'HOME_DRAW_RATE', 'HOME_LOSE_RATE', 'HOME_HOME_WIN_RATE', 'HOME_HOME_LOSE_RATE', 'HOME_HOME_DRAW_RATE', 'AWAY_WIN_RATE', 'AWAY_DRAW_RATE', 'AWAY_LOSE_RATE', 'AWAY_AWAY_WIN_RATE', 'AWAY_AWAY_LOSE_RATE', 'AWAY_AWAY_DRAW_RATE']]

df['OU35'] = [1 if x > 3.5 else 0 for x in df['TOTAL_GOAL_COUNT']]


df1 = df.drop(['MATCH_ID', 'MATCH_DATETIME', 'LEAGUE', 'HOME_TEAM',
               'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'TOTAL_GOAL_COUNT'], axis=1)


# Split train and test datasets
X = df1
X_train = df1.query('ML_TYPE == "TRAIN"')
X_val = df1.query('ML_TYPE == "VALID"')
X_test = df1.query('ML_TYPE == "TEST" | ML_TYPE == "PREDICT"')
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


model1 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    XGBClassifier(n_estimators=20, random_state=42, n_jobs=-1)
)

model1.fit(X_train, y_train)


print('1st XGBClassifier - Training Accuracy:', model1.score(X_train, y_train))
print('1st XGBClassifier - Validation Accuracy:', model1.score(X_val, y_val))
print('1st XGBClassifier - Test Accuracy:', model1.score(X_test, y_test))


model2 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(strategy='median'),
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)
model2.fit(X_train, y_train)

print('1st RandomForestClassifier - Training accuracy:',
      model2.score(X_train, y_train))
print('1st RandomForestClassifier - Validation accuracy:',
      model2.score(X_val, y_val))
print('1st RandomForestClassifier - Test accuracy:', model2.score(X_test, y_test))


model3 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    LogisticRegression(max_iter=300, random_state=42)
)

model3.fit(X_train, y_train)

print('1st LogisticRegression - Training Accuracy:',
      model3.score(X_train, y_train))
print('1st LogisticRegression - Validation Accuracy:', model3.score(X_val, y_val))
print('1st LogisticRegression - Test Accuracy:', model3.score(X_test, y_test))


# remove negative features
# X_train = X_train[['STR_A_AVG_A', 'STR_HDA_YINGWO_D', 'AWAY_AVG_GF', 'STR_A_MACAU_H', 'STR_A_AVG_H', 'GAME_POINT', 'STR_A_YINGWO_A', 'STR_O_YINGWO_HI_DIFF', 'STR_HDA_MACAU_H', 'STR_OU_BET365_HDC', 'STR_HDA_MACAU_D', 'STR_OU_PINNACLE_HDC', 'AWAY_AWAY_GA', 'AWAY_AWAY_GF', 'AWAY_WIN_RATE', 'AWAY_LOSE_RATE', 'AWAY_TOTAL_GA', 'STR_A_PINNACLE_A', 'STR_OU_YINGWO_HI', 'STR_HDA_PINNACLE_H', 'AWAY_AWAY_DRAW_RATE', 'STR_HDA_HKJC_H', 'STR_A_MACAU_A', 'HOME_DRAW_RATE', 'STR_O_YINGWO_LO_DIFF', 'STR_A_YINGWO_H', 'STR_HDA_HKJC_D', 'AWAY_DRAW_RATE', 'STR_A_MACAU_HDC', 'STR_O_MACAU_LO_DIFF',
#                    'HOME_AVG_GF', 'STR_HDA_BET365_H', 'STR_HDA_MACAU_A', 'STR_OU_AVG_HI', 'STR_OU_AVG_LO', 'STR_A_BET365_A', 'STR_OU_HKJC_HDC', 'STR_HDA_BET365_A', 'STR_O_HKJC_LO_DIFF', 'STR_O_HKJC_HI_DIFF', 'STR_OU_PINNACLE_LO', 'AWAY_AWAY_LOSE_RATE', 'HOME_HOME_DRAW_RATE', 'HOME_HOME_GF', 'STR_OU_MODE_HDC', 'STR_A_PINNACLE_HDC', 'HOME_ADV', 'AWAY_TOTAL_GF', 'AWAY_ADV', 'STR_A_MODE_HDC', 'STR_HDA_HKJC_A', 'STR_O_BET365_LO_DIFF', 'STR_OU_HKJC_LO', 'STR_OU_HKJC_HI', 'STR_OU_MACAU_HDC', 'STR_A_YINGWO_HDC', 'STR_A_PINNACLE_H', 'STR_OU_PINNACLE_HI', 'AWAY_AWAY_WIN_RATE', 'STR_O_PINNACLE_HI_DIFF']]
# X_val = X_val[['STR_A_AVG_A', 'STR_HDA_YINGWO_D', 'AWAY_AVG_GF', 'STR_A_MACAU_H', 'STR_A_AVG_H', 'GAME_POINT', 'STR_A_YINGWO_A', 'STR_O_YINGWO_HI_DIFF', 'STR_HDA_MACAU_H', 'STR_OU_BET365_HDC', 'STR_HDA_MACAU_D', 'STR_OU_PINNACLE_HDC', 'AWAY_AWAY_GA', 'AWAY_AWAY_GF', 'AWAY_WIN_RATE', 'AWAY_LOSE_RATE', 'AWAY_TOTAL_GA', 'STR_A_PINNACLE_A', 'STR_OU_YINGWO_HI', 'STR_HDA_PINNACLE_H', 'AWAY_AWAY_DRAW_RATE', 'STR_HDA_HKJC_H', 'STR_A_MACAU_A', 'HOME_DRAW_RATE', 'STR_O_YINGWO_LO_DIFF', 'STR_A_YINGWO_H', 'STR_HDA_HKJC_D', 'AWAY_DRAW_RATE', 'STR_A_MACAU_HDC', 'STR_O_MACAU_LO_DIFF',
#                'HOME_AVG_GF', 'STR_HDA_BET365_H', 'STR_HDA_MACAU_A', 'STR_OU_AVG_HI', 'STR_OU_AVG_LO', 'STR_A_BET365_A', 'STR_OU_HKJC_HDC', 'STR_HDA_BET365_A', 'STR_O_HKJC_LO_DIFF', 'STR_O_HKJC_HI_DIFF', 'STR_OU_PINNACLE_LO', 'AWAY_AWAY_LOSE_RATE', 'HOME_HOME_DRAW_RATE', 'HOME_HOME_GF', 'STR_OU_MODE_HDC', 'STR_A_PINNACLE_HDC', 'HOME_ADV', 'AWAY_TOTAL_GF', 'AWAY_ADV', 'STR_A_MODE_HDC', 'STR_HDA_HKJC_A', 'STR_O_BET365_LO_DIFF', 'STR_OU_HKJC_LO', 'STR_OU_HKJC_HI', 'STR_OU_MACAU_HDC', 'STR_A_YINGWO_HDC', 'STR_A_PINNACLE_H', 'STR_OU_PINNACLE_HI', 'AWAY_AWAY_WIN_RATE', 'STR_O_PINNACLE_HI_DIFF']]
# X_test = X_test[['STR_A_AVG_A', 'STR_HDA_YINGWO_D', 'AWAY_AVG_GF', 'STR_A_MACAU_H', 'STR_A_AVG_H', 'GAME_POINT', 'STR_A_YINGWO_A', 'STR_O_YINGWO_HI_DIFF', 'STR_HDA_MACAU_H', 'STR_OU_BET365_HDC', 'STR_HDA_MACAU_D', 'STR_OU_PINNACLE_HDC', 'AWAY_AWAY_GA', 'AWAY_AWAY_GF', 'AWAY_WIN_RATE', 'AWAY_LOSE_RATE', 'AWAY_TOTAL_GA', 'STR_A_PINNACLE_A', 'STR_OU_YINGWO_HI', 'STR_HDA_PINNACLE_H', 'AWAY_AWAY_DRAW_RATE', 'STR_HDA_HKJC_H', 'STR_A_MACAU_A', 'HOME_DRAW_RATE', 'STR_O_YINGWO_LO_DIFF', 'STR_A_YINGWO_H', 'STR_HDA_HKJC_D', 'AWAY_DRAW_RATE', 'STR_A_MACAU_HDC', 'STR_O_MACAU_LO_DIFF',
#                  'HOME_AVG_GF', 'STR_HDA_BET365_H', 'STR_HDA_MACAU_A', 'STR_OU_AVG_HI', 'STR_OU_AVG_LO', 'STR_A_BET365_A', 'STR_OU_HKJC_HDC', 'STR_HDA_BET365_A', 'STR_O_HKJC_LO_DIFF', 'STR_O_HKJC_HI_DIFF', 'STR_OU_PINNACLE_LO', 'AWAY_AWAY_LOSE_RATE', 'HOME_HOME_DRAW_RATE', 'HOME_HOME_GF', 'STR_OU_MODE_HDC', 'STR_A_PINNACLE_HDC', 'HOME_ADV', 'AWAY_TOTAL_GF', 'AWAY_ADV', 'STR_A_MODE_HDC', 'STR_HDA_HKJC_A', 'STR_O_BET365_LO_DIFF', 'STR_OU_HKJC_LO', 'STR_OU_HKJC_HI', 'STR_OU_MACAU_HDC', 'STR_A_YINGWO_HDC', 'STR_A_PINNACLE_H', 'STR_OU_PINNACLE_HI', 'AWAY_AWAY_WIN_RATE', 'STR_O_PINNACLE_HI_DIFF']]

X_train = X_train[['HOME_HOME_GF', 'STR_HDA_BET365_D', 'STR_HDA_MACAU_H', 'HOME_DRAW_RATE', 'HOME_ADV', 'STR_HDA_BET365_H', 'STR_O_PINNACLE_HI_DIFF', 'STR_A_BET365_A', 'STR_O_HKJC_HI_DIFF', 'STR_OU_YINGWO_LO', 'STR_O_HKJC_LO_DIFF', 'STR_HDA_PINNACLE_D', 'HOME_WIN_RATE', 'AWAY_TOTAL_GF', 'AWAY_TOTAL_GA', 'STR_A_YINGWO_HDC', 'STR_OU_HKJC_HDC', 'STR_A_MACAU_A',
                   'AWAY_AWAY_WIN_RATE', 'STR_HDA_YINGWO_A', 'STR_OU_YINGWO_HDC', 'STR_OU_BET365_LO', 'AWAY_AWAY_AVG_GF', 'STR_O_BET365_LO_DIFF', 'STR_HDA_HKJC_A', 'AWAY_AVG_GF', 'STR_OU_YINGWO_HI', 'STR_O_BET365_HI_DIFF', 'STR_OU_PINNACLE_HI', 'AWAY_DRAW_RATE', 'STR_A_YINGWO_H', 'STR_OU_MACAU_LO', 'AWAY_AWAY_GA', 'AWAY_WIN_RATE', 'STR_OU_MACAU_HI', 'HOME_TOTAL_GA', 'STR_OU_MACAU_HDC']]
X_val = X_val[['HOME_HOME_GF', 'STR_HDA_BET365_D', 'STR_HDA_MACAU_H', 'HOME_DRAW_RATE', 'HOME_ADV', 'STR_HDA_BET365_H', 'STR_O_PINNACLE_HI_DIFF', 'STR_A_BET365_A', 'STR_O_HKJC_HI_DIFF', 'STR_OU_YINGWO_LO', 'STR_O_HKJC_LO_DIFF', 'STR_HDA_PINNACLE_D', 'HOME_WIN_RATE', 'AWAY_TOTAL_GF', 'AWAY_TOTAL_GA', 'STR_A_YINGWO_HDC', 'STR_OU_HKJC_HDC', 'STR_A_MACAU_A',
               'AWAY_AWAY_WIN_RATE', 'STR_HDA_YINGWO_A', 'STR_OU_YINGWO_HDC', 'STR_OU_BET365_LO', 'AWAY_AWAY_AVG_GF', 'STR_O_BET365_LO_DIFF', 'STR_HDA_HKJC_A', 'AWAY_AVG_GF', 'STR_OU_YINGWO_HI', 'STR_O_BET365_HI_DIFF', 'STR_OU_PINNACLE_HI', 'AWAY_DRAW_RATE', 'STR_A_YINGWO_H', 'STR_OU_MACAU_LO', 'AWAY_AWAY_GA', 'AWAY_WIN_RATE', 'STR_OU_MACAU_HI', 'HOME_TOTAL_GA', 'STR_OU_MACAU_HDC']]
X_test = X_test[['HOME_HOME_GF', 'STR_HDA_BET365_D', 'STR_HDA_MACAU_H', 'HOME_DRAW_RATE', 'HOME_ADV', 'STR_HDA_BET365_H', 'STR_O_PINNACLE_HI_DIFF', 'STR_A_BET365_A', 'STR_O_HKJC_HI_DIFF', 'STR_OU_YINGWO_LO', 'STR_O_HKJC_LO_DIFF', 'STR_HDA_PINNACLE_D', 'HOME_WIN_RATE', 'AWAY_TOTAL_GF', 'AWAY_TOTAL_GA', 'STR_A_YINGWO_HDC', 'STR_OU_HKJC_HDC', 'STR_A_MACAU_A',
                 'AWAY_AWAY_WIN_RATE', 'STR_HDA_YINGWO_A', 'STR_OU_YINGWO_HDC', 'STR_OU_BET365_LO', 'AWAY_AWAY_AVG_GF', 'STR_O_BET365_LO_DIFF', 'STR_HDA_HKJC_A', 'AWAY_AVG_GF', 'STR_OU_YINGWO_HI', 'STR_O_BET365_HI_DIFF', 'STR_OU_PINNACLE_HI', 'AWAY_DRAW_RATE', 'STR_A_YINGWO_H', 'STR_OU_MACAU_LO', 'AWAY_AWAY_GA', 'AWAY_WIN_RATE', 'STR_OU_MACAU_HI', 'HOME_TOTAL_GA', 'STR_OU_MACAU_HDC']]


model1 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    XGBClassifier(n_estimators=12, random_state=42, n_jobs=-
                  1, learning_rate=0.097, subsample=1)
)

model1.fit(X_train, y_train)

print('2nd XGBClassifier - Training Accuracy:', model1.score(X_train, y_train))
print('2nd XGBClassifier - Validation Accuracy:', model1.score(X_val, y_val))
print('2nd XGBClassifier - Test Accuracy:', model1.score(X_test, y_test))


model2 = Pipeline([
                  ('oe', ce.OrdinalEncoder()),
                  ('impute', SimpleImputer(strategy='mean')),
                  ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
                  ])

model2.fit(X_train, y_train)

print('2nd RandomForestClassifier - Training accuracy:',
      model2.score(X_train, y_train))
print('2nd RandomForestClassifier - Validation accuracy:',
      model2.score(X_val, y_val))
print('2nd RandomForestClassifier - Test accuracy:', model2.score(X_test, y_test))


transformers = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(strategy='median')
)

X_train_transformed = transformers.fit_transform(X_train)
X_val_transformed = transformers.transform(X_val)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_transformed, y_train)


model3 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    LogisticRegression(random_state=42)
)

model3.fit(X_train, y_train)

print('2nd LogisticRegression - Training Accuracy:',
      model3.score(X_train, y_train))
print('2nd LogisticRegression - Validation Accuracy:', model3.score(X_val, y_val))
print('2nd LogisticRegression - Test Accuracy:', model3.score(X_test, y_test))


test = df[['MATCH_ID', 'MATCH_DATETIME', 'ML_TYPE', 'LEAGUE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'STR_OU_MACAU_HDC',
           'STR_OU_MACAU_HI', 'STR_OU_MACAU_LO', 'STR_OU_HKJC_HDC', 'STR_OU_HKJC_HI', 'STR_OU_HKJC_LO', 'TOTAL_GOAL_COUNT', ou_hdc]]

test = test.query('ML_TYPE == "TEST" | ML_TYPE == "PREDICT"')


y_pred = model1.predict(X_test)
class_probabilities = model1.predict_proba(X_test)

pred = pd.DataFrame(y_pred, columns=['pred'])
prob = pd.DataFrame(class_probabilities, columns=['prob0', 'prob1'])

test.reset_index(drop=True, inplace=True)
pred.reset_index(drop=True, inplace=True)
prob.reset_index(drop=True, inplace=True)

test_result = pd.concat([test, prob, pred], axis=1)


col_ml_type = '35'

# # Remove previous record
# sql = 'DELETE FROM ML_PREDICT_HILO WHERE ML_TYPE=:col_ml_type AND MATCH_DATE=:db_match_date'
# c.execute(sql, [col_ml_type, db_match_date])

gsheet_data = []
data_to_insert = []
# Insert db
for i, j in test_result.iterrows():
    col_match_id = int(test_result.loc[i, 'MATCH_ID'])
    col_match_datetime = str(test_result.loc[i, 'MATCH_DATETIME'])
    col_league = str(test_result.loc[i, 'LEAGUE'])
    col_home_team = str(test_result.loc[i, 'HOME_TEAM'])
    col_away_team = str(test_result.loc[i, 'AWAY_TEAM'])
    col_home_ft_goal = test_result.loc[i, 'HOME_FT_GOAL']
    if math.isnan(col_home_ft_goal):
        col_home_ft_goal = None
        col_away_ft_goal = None
        col_total_goal_count = None
    else:
        col_home_ft_goal = int(test_result.loc[i, 'HOME_FT_GOAL'])
        col_away_ft_goal = int(test_result.loc[i, 'AWAY_FT_GOAL'])
        col_total_goal_count = int(test_result.loc[i, 'TOTAL_GOAL_COUNT'])
    col_str_ou_macau_hdc = float(test_result.loc[i, 'STR_OU_MACAU_HDC'])
    col_str_ou_macau_hi = float(test_result.loc[i, 'STR_OU_MACAU_HI'])
    col_str_ou_macau_lo = float(test_result.loc[i, 'STR_OU_MACAU_LO'])
    col_str_ou_hkjc_hdc = float(test_result.loc[i, 'STR_OU_HKJC_HDC'])
    col_str_ou_hkjc_hi = float(test_result.loc[i, 'STR_OU_HKJC_HI'])
    col_str_ou_hkjc_lo = float(test_result.loc[i, 'STR_OU_HKJC_LO'])
    col_ou_hdc = int(test_result.loc[i, ou_hdc])
    col_prob0 = float(test_result.loc[i, 'prob0'])
    col_prob1 = float(test_result.loc[i, 'prob1'])
    col_pred = int(test_result.loc[i, 'pred'])
    data_to_insert.append([col_ml_type, col_match_id, col_match_datetime, col_league, col_home_team, col_away_team, col_str_ou_macau_hdc,
                           col_str_ou_macau_hi, col_str_ou_macau_lo, col_str_ou_hkjc_hdc, col_str_ou_hkjc_hi, col_str_ou_hkjc_lo, col_prob1, db_match_date])
    gsheet_data.append([col_match_id, col_match_datetime, col_ml_type, col_league, col_home_team, col_away_team, col_home_ft_goal, col_away_ft_goal, col_str_ou_macau_hdc,
                        col_str_ou_macau_hi, col_str_ou_macau_lo, col_str_ou_hkjc_hdc, col_str_ou_hkjc_hi, col_str_ou_hkjc_lo, col_total_goal_count, col_ou_hdc, col_prob0, col_prob1, col_pred])

# sql = 'INSERT INTO ML_PREDICT_HILO (ML_TYPE,MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,STR_OU_MACAU_HDC,STR_OU_MACAU_HI,STR_OU_MACAU_LO,STR_OU_HKJC_HDC,STR_OU_HKJC_HI,STR_OU_HKJC_LO,HI_PROB,MATCH_DATE) VALUES (:col_ml_type,:col_match_id,TO_DATE(:col_match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),:col_league,:col_home_team,:col_away_team,:col_str_ou_macau_hdc,:col_str_ou_macau_hi,:col_str_ou_macau_lo,:col_str_ou_hkjc_hdc,:col_str_ou_hkjc_hi,:col_str_ou_hkjc_lo,:col_prob1,:db_match_date)'
# c.executemany(sql, data_to_insert)
# connection.commit()

# Google sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "overunderml.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ML OverUnder").worksheet("35")

start_row = 4
end_row = sheet.row_count
if end_row >= start_row:
    sheet.delete_rows(start_row, end_row)

sheet.append_rows(gsheet_data)


# Prediction
col_ml_type = '35'
predict_result = test_result.query('ML_TYPE == "PREDICT"')

# Remove previous record
sql = 'DELETE FROM ML_PREDICT_HILO WHERE ML_TYPE=:col_ml_type AND MATCH_DATE=:db_match_date'
c.execute(sql, [col_ml_type, db_match_date])

data_to_insert = []
# Insert db
for i, j in predict_result.iterrows():
    col_match_id = int(predict_result.loc[i, 'MATCH_ID'])
    col_match_datetime = str(predict_result.loc[i, 'MATCH_DATETIME'])
    col_league = str(predict_result.loc[i, 'LEAGUE'])
    col_home_team = str(predict_result.loc[i, 'HOME_TEAM'])
    col_away_team = str(predict_result.loc[i, 'AWAY_TEAM'])
    col_str_ou_macau_hdc = float(predict_result.loc[i, 'STR_OU_MACAU_HDC'])
    col_str_ou_macau_hi = float(predict_result.loc[i, 'STR_OU_MACAU_HI'])
    col_str_ou_macau_lo = float(predict_result.loc[i, 'STR_OU_MACAU_LO'])
    col_str_ou_hkjc_hdc = float(predict_result.loc[i, 'STR_OU_HKJC_HDC'])
    col_str_ou_hkjc_hi = float(predict_result.loc[i, 'STR_OU_HKJC_HI'])
    col_str_ou_hkjc_lo = float(predict_result.loc[i, 'STR_OU_HKJC_LO'])
    col_prob1 = float(predict_result.loc[i, 'prob1'])
    data_to_insert.append([col_ml_type, col_match_id, col_match_datetime, col_league, col_home_team, col_away_team, col_str_ou_macau_hdc,
                           col_str_ou_macau_hi, col_str_ou_macau_lo, col_str_ou_hkjc_hdc, col_str_ou_hkjc_hi, col_str_ou_hkjc_lo, col_prob1, db_match_date])

sql = 'INSERT INTO ML_PREDICT_HILO (ML_TYPE,MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,STR_OU_MACAU_HDC,STR_OU_MACAU_HI,STR_OU_MACAU_LO,STR_OU_HKJC_HDC,STR_OU_HKJC_HI,STR_OU_HKJC_LO,HI_PROB,MATCH_DATE) VALUES (:col_ml_type,:col_match_id,TO_DATE(:col_match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),:col_league,:col_home_team,:col_away_team,:col_str_ou_macau_hdc,:col_str_ou_macau_hi,:col_str_ou_macau_lo,:col_str_ou_hkjc_hdc,:col_str_ou_hkjc_hi,:col_str_ou_hkjc_lo,:col_prob1,:db_match_date)'
c.executemany(sql, data_to_insert)
connection.commit()
