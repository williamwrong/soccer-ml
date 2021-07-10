#!/usr/bin/env python
# coding: utf-8


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
csv_name = 'str_hilo_253_new.csv'
output_file = '%s\\data\\%s' % (curr_path, csv_name)
error_file = '%s\\error.txt' % (curr_path)
if os.path.exists(output_file):
    os.remove(output_file)
if os.path.exists(error_file):
    os.remove(error_file)

ou_hdc = 'OU25'

# Match date
current_match_date = datetime.now()
current_time = datetime.now().time()
if current_time >= time(0, 0) and current_time <= time(11, 30):
    previous_day = current_match_date - timedelta(days=1)
    current_match_date = previous_day
db_match_date = current_match_date.strftime("%Y%m%d")
current_match_date = current_match_date.strftime("%Y-%m-%d 11:30:00")

print('Execution time: %s' % strftime('%Y-%m-%d %H:%M:%S'))


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
    file = codecs.open(output_file, "a+", "utf-8")

    # write file header
    file.write('MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,HOME_FT_GOAL,AWAY_FT_GOAL,TOTAL_GOAL_COUNT,ML_TYPE,STR_OU_MODE_HDC,STR_OU_MACAU_HDC,STR_OU_BET365_HDC,STR_OU_YINGYO_HDC,STR_OU_PINGBOK_HDC,STR_OU_HKJC_HDC,STR_OU_AVG_HI,STR_OU_MEDIAN_HI,STR_OU_MACAU_HI,STR_OU_BET365_HI,STR_OU_YINGWO_HI,STR_OU_PINGBOK_HI,STR_OU_HKJC_HI,STR_OU_MACAU_HI_DIFF,STR_OU_BET365_HI_DIFF,STR_OU_YINGWO_HI_DIFF,STR_OU_PINGBOK_HI_DIFF,STR_OU_HKJC_HI_DIFF,STR_OU_AVG_LO,STR_OU_MEDIAN_LO,STR_OU_MACAU_LO,STR_OU_BET365_LO,STR_OU_YINGWO_LO,STR_OU_PINGBOK_LO,STR_OU_HKJC_LO,STR_OU_MACAU_LO_DIFF,STR_OU_BET365_LO_DIFF,STR_OU_YINGWO_LO_DIFF,STR_OU_PINGBOK_LO_DIFF,STR_OU_HKJC_LO_DIFF,HOME_TOTAL_GF,HOME_TOTAL_GA,HOME_AVG_GF,HOME_HOME_GF,HOME_HOME_GA,HOME_HOME_AVG_GF,AWAY_TOTAL_GF,AWAY_TOTAL_GA,AWAY_AVG_GF,AWAY_AWAY_GF,AWAY_AWAY_GA,AWAY_AWAY_AVG_GF,HOME_ADV,AWAY_ADV,GAME_POINT,STR_MACAU_H,STR_BET365_H,STR_YINGWO_H,STR_PINNACLE_H,STR_HKJC_H,STR_MACAU_D,STR_BET365_D,STR_YINGWO_D,STR_PINNACLE_D,STR_HKJC_D,STR_MACAU_A,STR_BET365_A,STR_YINGWO_A,STR_PINNACLE_A,STR_HKJC_A\n')
    sql = """
        SELECT
            info.MATCH_ID, info.MATCH_DATETIME, info.LEAGUE, info.HOME_TEAM, info.AWAY_TEAM, info.HOME_FT_GOAL, info.AWAY_FT_GOAL, info.HOME_FT_GOAL+info.AWAY_FT_GOAL AS TOTAL_GOAL_COUNT, 
            CASE 
                WHEN info.MATCH_DATETIME < TO_TIMESTAMP('2020-06-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'TRAIN'
                WHEN info.MATCH_DATETIME >= TO_TIMESTAMP('2020-06-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') AND info.MATCH_DATETIME < TO_TIMESTAMP('2020-09-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 'VALID'
                WHEN info.MATCH_DATETIME >= TO_TIMESTAMP('2020-09-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') AND info.MATCH_DATETIME < TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') THEN 'TEST'
                WHEN info.MATCH_DATETIME >= TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') THEN 'PREDICT'
            END AS ML_TYPE, 
            hilo.STR_MODE_HDC AS STR_OU_MODE_HDC, hilo.STR_MACAU_HDC AS STR_OU_MACAU_HDC, hilo.STR_BET365_HDC AS STR_OU_BET365_HDC, hilo.STR_YINGYO_HDC AS STR_OU_YINGYO_HDC, hilo.STR_PINGBOK_HDC AS STR_OU_PINGBOK_HDC, hilo.STR_HKJC_HDC AS STR_OU_HKJC_HDC, 
            ROUND(hilo.STR_O_AVG_HI,4) AS STR_OU_AVG_HI, hilo.STR_O_MEDIAN_HI AS STR_OU_MEDIAN_HI, hilo.STR_O_MACAU_HI AS STR_OU_MACAU_HI, hilo.STR_O_BET365_HI AS STR_OU_BET365_HI, hilo.STR_O_YINGYO_HI AS STR_OU_YINGWO_HI, hilo.STR_O_PINGBOK_HI AS STR_OU_PINGBOK_HI, hilo.STR_O_HKJC_HI AS STR_OU_HKJC_HI, 
            ROUND((hilo.STR_O_MACAU_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_OU_MACAU_HI_DIFF, ROUND((hilo.STR_O_BET365_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_OU_BET365_HI_DIFF, ROUND((hilo.STR_O_YINGYO_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_OU_YINGWO_HI_DIFF, ROUND((hilo.STR_O_PINGBOK_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_OU_PINGBOK_HI_DIFF, ROUND((hilo.STR_O_HKJC_HI-hilo.STR_O_AVG_HI)/hilo.STR_O_AVG_HI,4) AS STR_OU_HKJC_HI_DIFF, 
            ROUND(hilo.STR_O_AVG_LO,4) AS STR_OU_AVG_LO, hilo.STR_O_MEDIAN_LO AS STR_OU_MEDIAN_LO, hilo.STR_O_MACAU_LO AS STR_OU_MACAU_LO, hilo.STR_O_BET365_LO AS STR_OU_BET365_LO, hilo.STR_O_YINGYO_LO AS STR_OU_YINGWO_LO, hilo.STR_O_PINGBOK_LO AS STR_OU_PINGBOK_LO, hilo.STR_O_HKJC_LO AS STR_OU_HKJC_LO, 
            ROUND((hilo.STR_O_MACAU_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_OU_MACAU_LO_DIFF, ROUND((hilo.STR_O_BET365_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_OU_BET365_LO_DIFF, ROUND((hilo.STR_O_YINGYO_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_OU_YINGWO_LO_DIFF, ROUND((hilo.STR_O_PINGBOK_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_OU_PINGBOK_LO_DIFF, ROUND((hilo.STR_O_HKJC_LO-hilo.STR_O_AVG_LO)/hilo.STR_O_AVG_LO,4) AS STR_OU_HKJC_LO_DIFF, 
            recent.HOME_TOTAL_GF, recent.HOME_TOTAL_GA, ROUND(recent.HOME_AVG_GF,4) AS HOME_AVG_GF, recent.HOME_HOME_GF, recent.HOME_HOME_GA, ROUND(recent.HOME_HOME_AVG_GF,4) AS HOME_HOME_AVG_GF, 
            recent.AWAY_TOTAL_GF, recent.AWAY_TOTAL_GA, ROUND(recent.AWAY_AVG_GF,4) AS AWAY_AVG_GF, recent.AWAY_AWAY_GF, recent.AWAY_AWAY_GA, ROUND(recent.AWAY_AWAY_AVG_GF,4) AS AWAY_AWAY_AVG_GF, 
            (recent.HOME_HOME_GF+recent.AWAY_AWAY_GA)/10 AS HOME_ADV, (recent.HOME_HOME_GA+recent.AWAY_AWAY_GF)/10 AS AWAY_ADV, (recent.HOME_HOME_GF+recent.AWAY_AWAY_GA)/10+(recent.HOME_HOME_GA+recent.AWAY_AWAY_GF)/10 AS GAME_POINT, 
            macau.HOME_ODD AS STR_MACAU_H, bet365.HOME_ODD AS STR_BET365_H, yingwo.HOME_ODD AS STR_YINGWO_H, pinnacle.HOME_ODD AS STR_PINNACLE_H, hkjc.HOME_ODD AS STR_HKJC_H, 
            macau.DRAW_ODD AS STR_MACAU_D, bet365.DRAW_ODD AS STR_BET365_D, yingwo.DRAW_ODD AS STR_YINGWO_D, pinnacle.DRAW_ODD AS STR_PINNACLE_D, hkjc.DRAW_ODD AS STR_HKJC_D, 
            macau.AWAY_ODD AS STR_MACAU_A, bet365.AWAY_ODD AS STR_BET365_A, yingwo.AWAY_ODD AS STR_YINGWO_A, pinnacle.AWAY_ODD AS STR_PINNACLE_A, hkjc.AWAY_ODD AS STR_HKJC_A 
        FROM
            HILO_MERGE2 hilo, RECENT_RAW recent, MATCH_INFO info, HDA_RAW macau, HDA_RAW bet365, HDA_RAW yingwo, HDA_RAW pinnacle, HDA_RAW hkjc
        WHERE
            info.MATCH_ID=hilo.MATCH_ID AND info.MATCH_ID=recent.MATCH_ID AND info.MATCH_ID=macau.MATCH_ID AND info.MATCH_ID=bet365.MATCH_ID AND info.MATCH_ID=yingwo.MATCH_ID AND info.MATCH_ID=pinnacle.MATCH_ID AND info.MATCH_ID=hkjc.MATCH_ID
            AND macau.BOOKMAKER='澳门' AND bet365.BOOKMAKER='bet365' AND yingwo.BOOKMAKER='盈禾' AND pinnacle.BOOKMAKER='Pinnacle' AND hkjc.BOOKMAKER='香港马会'
            AND macau.HANDICAP_TYPE=0 AND bet365.HANDICAP_TYPE=0 AND yingwo.HANDICAP_TYPE=0 AND pinnacle.HANDICAP_TYPE=0 AND hkjc.HANDICAP_TYPE=0
            AND hilo.STR_MODE_HDC IS NOT NULL AND hilo.STR_MACAU_HDC IS NOT NULL AND hilo.STR_BET365_HDC IS NOT NULL AND hilo.STR_YINGYO_HDC IS NOT NULL AND hilo.STR_PINGBOK_HDC IS NOT NULL AND hilo.STR_HKJC_HDC IS NOT NULL 
            AND ROUND(hilo.STR_O_AVG_HI,4) IS NOT NULL AND hilo.STR_O_MEDIAN_HI IS NOT NULL AND hilo.STR_O_MACAU_HI IS NOT NULL AND hilo.STR_O_BET365_HI IS NOT NULL AND hilo.STR_O_YINGYO_HI IS NOT NULL AND hilo.STR_O_PINGBOK_HI IS NOT NULL AND hilo.STR_O_HKJC_HI IS NOT NULL 
            AND ROUND(hilo.STR_O_AVG_LO,4) IS NOT NULL AND hilo.STR_O_MEDIAN_LO IS NOT NULL AND hilo.STR_O_MACAU_LO IS NOT NULL AND hilo.STR_O_BET365_LO IS NOT NULL AND hilo.STR_O_YINGYO_LO IS NOT NULL AND hilo.STR_O_PINGBOK_LO IS NOT NULL AND hilo.STR_O_HKJC_LO IS NOT NULL 
            AND recent.HOME_TOTAL_GF IS NOT NULL AND recent.HOME_TOTAL_GA IS NOT NULL AND ROUND(recent.HOME_AVG_GF,4) IS NOT NULL AND recent.HOME_HOME_GF IS NOT NULL AND recent.HOME_HOME_GA IS NOT NULL AND ROUND(recent.HOME_HOME_AVG_GF,4) IS NOT NULL 
            AND recent.AWAY_TOTAL_GF IS NOT NULL AND recent.AWAY_TOTAL_GA IS NOT NULL AND ROUND(recent.AWAY_AVG_GF,4) IS NOT NULL AND recent.AWAY_AWAY_GF IS NOT NULL AND recent.AWAY_AWAY_GA IS NOT NULL AND ROUND(recent.AWAY_AWAY_AVG_GF,4) IS NOT NULL 
            AND (recent.HOME_HOME_GF+recent.AWAY_AWAY_GA)/10 IS NOT NULL AND (recent.HOME_HOME_GA+recent.AWAY_AWAY_GF)/10 IS NOT NULL AND (recent.HOME_HOME_GF+recent.AWAY_AWAY_GA)/10+(recent.HOME_HOME_GA+recent.AWAY_AWAY_GF)/10 IS NOT NULL 
            AND macau.HOME_ODD IS NOT NULL AND bet365.HOME_ODD IS NOT NULL AND yingwo.HOME_ODD IS NOT NULL AND pinnacle.HOME_ODD IS NOT NULL AND hkjc.HOME_ODD IS NOT NULL 
            AND macau.DRAW_ODD IS NOT NULL AND bet365.DRAW_ODD IS NOT NULL AND yingwo.DRAW_ODD IS NOT NULL AND pinnacle.DRAW_ODD IS NOT NULL AND hkjc.DRAW_ODD IS NOT NULL 
            AND macau.AWAY_ODD IS NOT NULL AND bet365.AWAY_ODD IS NOT NULL AND yingwo.AWAY_ODD IS NOT NULL AND pinnacle.AWAY_ODD IS NOT NULL AND hkjc.AWAY_ODD IS NOT NULL 
            AND (info.MATCH_DATETIME >= TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') OR (info.MATCH_DATETIME < TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS') AND info.HOME_FT_GOAL IS NOT NULL))
        ORDER BY info.MATCH_DATETIME
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
        file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (row[0], row[1], row[2], row[3], row[4], ft_home_goal, ft_away_goal, total_goal, row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[
                   21], row[22], row[23], row[24], row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35], row[36], row[37], row[38], row[39], row[40], row[41], row[42], row[43], row[44], row[45], row[46], row[47], row[48], row[49], row[50], row[51], row[52], row[53], row[54], row[55], row[56], row[57], row[58], row[59], row[60], row[61], row[62], row[63], row[64], row[65], row[66], row[67], row[68]))

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


# #final feature selection
df = df[['MATCH_ID', 'MATCH_DATETIME', 'LEAGUE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'TOTAL_GOAL_COUNT', 'ML_TYPE', 'STR_OU_MODE_HDC', 'STR_OU_MACAU_HDC', 'STR_OU_BET365_HDC', 'STR_OU_YINGYO_HDC', 'STR_OU_PINGBOK_HDC', 'STR_OU_HKJC_HDC', 'STR_OU_AVG_HI', 'STR_OU_MEDIAN_HI', 'STR_OU_MACAU_HI', 'STR_OU_BET365_HI', 'STR_OU_YINGWO_HI', 'STR_OU_PINGBOK_HI', 'STR_OU_HKJC_HI', 'STR_OU_MACAU_HI_DIFF', 'STR_OU_BET365_HI_DIFF', 'STR_OU_YINGWO_HI_DIFF', 'STR_OU_PINGBOK_HI_DIFF', 'STR_OU_HKJC_HI_DIFF', 'STR_OU_AVG_LO', 'STR_OU_MEDIAN_LO', 'STR_OU_MACAU_LO', 'STR_OU_BET365_LO', 'STR_OU_YINGWO_LO', 'STR_OU_PINGBOK_LO',
         'STR_OU_HKJC_LO', 'STR_OU_MACAU_LO_DIFF', 'STR_OU_BET365_LO_DIFF', 'STR_OU_YINGWO_LO_DIFF', 'STR_OU_PINGBOK_LO_DIFF', 'STR_OU_HKJC_LO_DIFF', 'HOME_TOTAL_GF', 'HOME_TOTAL_GA', 'HOME_AVG_GF', 'HOME_HOME_GF', 'HOME_HOME_GA', 'HOME_HOME_AVG_GF', 'AWAY_TOTAL_GF', 'AWAY_TOTAL_GA', 'AWAY_AVG_GF', 'AWAY_AWAY_GF', 'AWAY_AWAY_GA', 'AWAY_AWAY_AVG_GF', 'HOME_ADV', 'AWAY_ADV', 'GAME_POINT', 'STR_MACAU_H', 'STR_BET365_H', 'STR_YINGWO_H', 'STR_PINNACLE_H', 'STR_HKJC_H', 'STR_MACAU_D', 'STR_BET365_D', 'STR_YINGWO_D', 'STR_PINNACLE_D', 'STR_HKJC_D', 'STR_MACAU_A', 'STR_BET365_A', 'STR_YINGWO_A', 'STR_PINNACLE_A', 'STR_HKJC_A']]

df['OU25'] = [1 if x > 2.5 else 0 for x in df['TOTAL_GOAL_COUNT']]


df1 = df.drop(['MATCH_ID', 'MATCH_DATETIME', 'LEAGUE', 'HOME_TEAM',
               'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'TOTAL_GOAL_COUNT'], axis=1)


# Split train and test datasets

X = df1
X_train = df1.query('ML_TYPE == "TRAIN"')
X_val = df1.query('ML_TYPE == "VALID"')
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

print('RandomForestClassifier - Training accuracy:',
      model2.score(X_train, y_train))
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


# remove negative features
X_train = X_train[['STR_HKJC_D', 'STR_OU_MODE_HDC', 'STR_OU_MACAU_HDC', 'AWAY_AVG_GF', 'STR_OU_HKJC_LO_DIFF', 'STR_OU_HKJC_HI_DIFF', 'STR_BET365_D', 'STR_OU_PINGBOK_LO_DIFF', 'STR_PINNACLE_D', 'STR_YINGWO_A', 'STR_MACAU_D', 'STR_OU_BET365_HDC', 'STR_OU_HKJC_LO', 'AWAY_AWAY_GF', 'STR_OU_BET365_LO_DIFF', 'STR_YINGWO_H', 'STR_BET365_H', 'HOME_TOTAL_GF', 'STR_OU_YINGYO_HDC', 'STR_HKJC_A', 'AWAY_TOTAL_GF', 'STR_OU_PINGBOK_HI',
                   'STR_BET365_A', 'STR_OU_YINGWO_HI', 'AWAY_TOTAL_GA', 'STR_OU_PINGBOK_LO', 'STR_OU_PINGBOK_HDC', 'STR_OU_MACAU_HI_DIFF', 'AWAY_ADV', 'STR_OU_MEDIAN_LO', 'STR_OU_AVG_HI', 'STR_OU_YINGWO_LO_DIFF', 'STR_OU_HKJC_HI', 'STR_PINNACLE_H', 'STR_OU_HKJC_HDC', 'STR_OU_MACAU_LO_DIFF', 'STR_MACAU_A', 'HOME_TOTAL_GA', 'STR_HKJC_H', 'AWAY_AWAY_GA', 'HOME_ADV', 'HOME_HOME_GF', 'HOME_HOME_AVG_GF', 'STR_OU_YINGWO_HI_DIFF', 'STR_OU_BET365_HI']]
X_val = X_val[['STR_HKJC_D', 'STR_OU_MODE_HDC', 'STR_OU_MACAU_HDC', 'AWAY_AVG_GF', 'STR_OU_HKJC_LO_DIFF', 'STR_OU_HKJC_HI_DIFF', 'STR_BET365_D', 'STR_OU_PINGBOK_LO_DIFF', 'STR_PINNACLE_D', 'STR_YINGWO_A', 'STR_MACAU_D', 'STR_OU_BET365_HDC', 'STR_OU_HKJC_LO', 'AWAY_AWAY_GF', 'STR_OU_BET365_LO_DIFF', 'STR_YINGWO_H', 'STR_BET365_H', 'HOME_TOTAL_GF', 'STR_OU_YINGYO_HDC', 'STR_HKJC_A', 'AWAY_TOTAL_GF', 'STR_OU_PINGBOK_HI',
               'STR_BET365_A', 'STR_OU_YINGWO_HI', 'AWAY_TOTAL_GA', 'STR_OU_PINGBOK_LO', 'STR_OU_PINGBOK_HDC', 'STR_OU_MACAU_HI_DIFF', 'AWAY_ADV', 'STR_OU_MEDIAN_LO', 'STR_OU_AVG_HI', 'STR_OU_YINGWO_LO_DIFF', 'STR_OU_HKJC_HI', 'STR_PINNACLE_H', 'STR_OU_HKJC_HDC', 'STR_OU_MACAU_LO_DIFF', 'STR_MACAU_A', 'HOME_TOTAL_GA', 'STR_HKJC_H', 'AWAY_AWAY_GA', 'HOME_ADV', 'HOME_HOME_GF', 'HOME_HOME_AVG_GF', 'STR_OU_YINGWO_HI_DIFF', 'STR_OU_BET365_HI']]
X_test = X_test[['STR_HKJC_D', 'STR_OU_MODE_HDC', 'STR_OU_MACAU_HDC', 'AWAY_AVG_GF', 'STR_OU_HKJC_LO_DIFF', 'STR_OU_HKJC_HI_DIFF', 'STR_BET365_D', 'STR_OU_PINGBOK_LO_DIFF', 'STR_PINNACLE_D', 'STR_YINGWO_A', 'STR_MACAU_D', 'STR_OU_BET365_HDC', 'STR_OU_HKJC_LO', 'AWAY_AWAY_GF', 'STR_OU_BET365_LO_DIFF', 'STR_YINGWO_H', 'STR_BET365_H', 'HOME_TOTAL_GF', 'STR_OU_YINGYO_HDC', 'STR_HKJC_A', 'AWAY_TOTAL_GF', 'STR_OU_PINGBOK_HI',
                 'STR_BET365_A', 'STR_OU_YINGWO_HI', 'AWAY_TOTAL_GA', 'STR_OU_PINGBOK_LO', 'STR_OU_PINGBOK_HDC', 'STR_OU_MACAU_HI_DIFF', 'AWAY_ADV', 'STR_OU_MEDIAN_LO', 'STR_OU_AVG_HI', 'STR_OU_YINGWO_LO_DIFF', 'STR_OU_HKJC_HI', 'STR_PINNACLE_H', 'STR_OU_HKJC_HDC', 'STR_OU_MACAU_LO_DIFF', 'STR_MACAU_A', 'HOME_TOTAL_GA', 'STR_HKJC_H', 'AWAY_AWAY_GA', 'HOME_ADV', 'HOME_HOME_GF', 'HOME_HOME_AVG_GF', 'STR_OU_YINGWO_HI_DIFF', 'STR_OU_BET365_HI']]


model1 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    XGBClassifier(n_estimators=12, random_state=42, n_jobs=-
                  1, learning_rate=0.097, subsample=1)
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

print('RandomForestClassifier - Training accuracy:',
      model2.score(X_train, y_train))
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


model3 = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    LogisticRegression(random_state=42)
)

model3.fit(X_train, y_train)

print('LogisticRegression - Training Accuracy:', model3.score(X_train, y_train))
print('LogisticRegression - Validation Accuracy:', model3.score(X_val, y_val))
print('LogisticRegression - Test Accuracy:', model3.score(X_test, y_test))


final = df[['MATCH_ID', 'MATCH_DATETIME', 'ML_TYPE', 'LEAGUE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'STR_OU_MACAU_HDC',
            'STR_OU_MACAU_HI', 'STR_OU_MACAU_LO', 'STR_OU_HKJC_HDC', 'STR_OU_HKJC_HI', 'STR_OU_HKJC_LO', 'TOTAL_GOAL_COUNT', ou_hdc]]

final = final.query('ML_TYPE == "TEST"')


y_pred = model1.predict(X_test)
class_probabilities = model1.predict_proba(X_test)

pred = pd.DataFrame(y_pred, columns=['pred'])
prob = pd.DataFrame(class_probabilities, columns=['prob0', 'prob1'])

final.reset_index(drop=True, inplace=True)
pred.reset_index(drop=True, inplace=True)
prob.reset_index(drop=True, inplace=True)

fin = pd.concat([final, prob, pred], axis=1)


# hi_prob = 0.6
# print('MATCH_DATETIME > \'%s\'' % current_match_date)
# finq = fin.query('MATCH_DATETIME > \'%s\' & prob1>=%s' % (current_match_date, hi_prob))

# # finq.tail(20)
# # fin.tail(50)
# # fin.head(50)
# finq = finq.drop(['HOME_FT_GOAL','AWAY_FT_GOAL','TOTAL_GOAL_COUNT','OU25','prob0','Correct','PL 0.55','PL 0.6','PL 0.65','PL 0.7'], axis=1)
# finq

# fin

col_ml_type = '253NEW'

# Remove previous record
sql = 'DELETE FROM ML_PREDICT_HILO WHERE ML_TYPE=:col_ml_type AND MATCH_DATE=:db_match_date'
c.execute(sql, [col_ml_type, db_match_date])

data_to_insert = []
# Insert db
for i, j in fin.iterrows():
    col_match_id = int(fin.loc[i, 'MATCH_ID'])
    col_match_datetime = str(fin.loc[i, 'MATCH_DATETIME'])
    col_league = str(fin.loc[i, 'LEAGUE'])
    col_home_team = str(fin.loc[i, 'HOME_TEAM'])
    col_away_team = str(fin.loc[i, 'AWAY_TEAM'])
    col_str_ou_macau_hdc = float(fin.loc[i, 'STR_OU_MACAU_HDC'])
    col_str_ou_macau_hi = float(fin.loc[i, 'STR_OU_MACAU_HI'])
    col_str_ou_macau_lo = float(fin.loc[i, 'STR_OU_MACAU_LO'])
    col_str_ou_hkjc_hdc = float(fin.loc[i, 'STR_OU_HKJC_HDC'])
    col_str_ou_hkjc_hi = float(fin.loc[i, 'STR_OU_HKJC_HI'])
    col_str_ou_hkjc_lo = float(fin.loc[i, 'STR_OU_HKJC_LO'])
    col_prob1 = float(fin.loc[i, 'prob1'])
    data_to_insert.append([col_ml_type, col_match_id, col_match_datetime, col_league, col_home_team, col_away_team, col_str_ou_macau_hdc,
                           col_str_ou_macau_hi, col_str_ou_macau_lo, col_str_ou_hkjc_hdc, col_str_ou_hkjc_hi, col_str_ou_hkjc_lo, col_prob1, db_match_date])

sql = 'INSERT INTO ML_PREDICT_HILO (ML_TYPE,MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,STR_OU_MACAU_HDC,STR_OU_MACAU_HI,STR_OU_MACAU_LO,STR_OU_HKJC_HDC,STR_OU_HKJC_HI,STR_OU_HKJC_LO,HI_PROB,MATCH_DATE) VALUES (:col_ml_type,:col_match_id,TO_DATE(:col_match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),:col_league,:col_home_team,:col_away_team,:col_str_ou_macau_hdc,:col_str_ou_macau_hi,:col_str_ou_macau_lo,:col_str_ou_hkjc_hdc,:col_str_ou_hkjc_hi,:col_str_ou_hkjc_lo,:col_prob1,:db_match_date)'
c.executemany(sql, data_to_insert)
connection.commit()


final = df[['MATCH_ID', 'MATCH_DATETIME', 'ML_TYPE', 'LEAGUE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_FT_GOAL', 'AWAY_FT_GOAL', 'STR_OU_MACAU_HDC',
            'STR_OU_MACAU_HI', 'STR_OU_MACAU_LO', 'STR_OU_HKJC_HDC', 'STR_OU_HKJC_HI', 'STR_OU_HKJC_LO', 'TOTAL_GOAL_COUNT', ou_hdc]]

final = final.query('ML_TYPE == "PREDICT"')

y_pred = model1.predict(X_test)
class_probabilities = model1.predict_proba(X_test)

pred = pd.DataFrame(y_pred, columns=['pred'])
prob = pd.DataFrame(class_probabilities, columns=['prob0', 'prob1'])

final.reset_index(drop=True, inplace=True)
pred.reset_index(drop=True, inplace=True)
prob.reset_index(drop=True, inplace=True)

fin = pd.concat([final, prob, pred], axis=1)
fin = fin.query('MATCH_ID.notnull()')
# fin


col_ml_type = '253NEW'

# Remove previous record
# sql = 'DELETE FROM ML_PREDICT_HILO WHERE ML_TYPE=:col_ml_type AND MATCH_DATE=:db_match_date'
# c.execute(sql, [col_ml_type, db_match_date])

data_to_insert = []
# Insert db
for i, j in fin.iterrows():
    col_match_id = int(fin.loc[i, 'MATCH_ID'])
    col_match_datetime = str(fin.loc[i, 'MATCH_DATETIME'])
    col_league = str(fin.loc[i, 'LEAGUE'])
    col_home_team = str(fin.loc[i, 'HOME_TEAM'])
    col_away_team = str(fin.loc[i, 'AWAY_TEAM'])
    col_str_ou_macau_hdc = float(fin.loc[i, 'STR_OU_MACAU_HDC'])
    col_str_ou_macau_hi = float(fin.loc[i, 'STR_OU_MACAU_HI'])
    col_str_ou_macau_lo = float(fin.loc[i, 'STR_OU_MACAU_LO'])
    col_str_ou_hkjc_hdc = float(fin.loc[i, 'STR_OU_HKJC_HDC'])
    col_str_ou_hkjc_hi = float(fin.loc[i, 'STR_OU_HKJC_HI'])
    col_str_ou_hkjc_lo = float(fin.loc[i, 'STR_OU_HKJC_LO'])
    col_prob1 = float(fin.loc[i, 'prob1'])
    data_to_insert.append([col_ml_type, col_match_id, col_match_datetime, col_league, col_home_team, col_away_team, col_str_ou_macau_hdc,
                           col_str_ou_macau_hi, col_str_ou_macau_lo, col_str_ou_hkjc_hdc, col_str_ou_hkjc_hi, col_str_ou_hkjc_lo, col_prob1, db_match_date])

sql = 'INSERT INTO ML_PREDICT_HILO (ML_TYPE,MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,STR_OU_MACAU_HDC,STR_OU_MACAU_HI,STR_OU_MACAU_LO,STR_OU_HKJC_HDC,STR_OU_HKJC_HI,STR_OU_HKJC_LO,HI_PROB,MATCH_DATE) VALUES (:col_ml_type,:col_match_id,TO_DATE(:col_match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),:col_league,:col_home_team,:col_away_team,:col_str_ou_macau_hdc,:col_str_ou_macau_hi,:col_str_ou_macau_lo,:col_str_ou_hkjc_hdc,:col_str_ou_hkjc_hi,:col_str_ou_hkjc_lo,:col_prob1,:db_match_date)'
c.executemany(sql, data_to_insert)
connection.commit()
