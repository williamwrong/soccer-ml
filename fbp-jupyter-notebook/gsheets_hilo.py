from oauth2client.service_account import ServiceAccountCredentials
import gspread
from time import strftime
import cx_Oracle
import math

db_user = 'JW'
db_password = '901203'
db_dsn = 'HOME-PC/XE'
db_encoding = 'UTF-8'

# Google sheets
# from pprint import pprint

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "overunderml.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ML OverUnder").sheet1

start_row = 4
end_row = sheet.row_count
if end_row >= start_row:
    sheet.delete_rows(start_row, end_row)


# Database connection
connection = None
try:
    connection = cx_Oracle.connect(
        db_user,
        db_password,
        db_dsn,
        encoding=db_encoding)

    c = connection.cursor()

    data_row = []

    hi_prob_35 = 0.60
    hi_prob_253 = 0.66
    hi_prob_5630 = 0.62
    hi_prob_logreg = 0.62
    hi_prob_twist = 0.65
    hi_prob_twist2 = 0.58
    hi_prob_hkjc = 0.53

    sql = """SELECT 
                a.ML_TYPE, a.MATCH_ID, a.MATCH_DATETIME, a.LEAGUE, a.HOME_TEAM, a.AWAY_TEAM, b.HOME_FT_GOAL, b.AWAY_FT_GOAL, a.STR_OU_MACAU_HDC, a.STR_OU_MACAU_HI, a.STR_OU_MACAU_LO, a.STR_OU_HKJC_HDC, a.STR_OU_HKJC_HI, a.STR_OU_HKJC_LO, a.HI_PROB, a.MATCH_DATE 
            FROM 
                ML_PREDICT_HILO a, MATCH_INFO b
            WHERE 
                a.MATCH_ID=b.MATCH_ID
                AND (
                    (a.ML_TYPE='35' AND a.HI_PROB>=%s)
                    OR (a.ML_TYPE='253' AND a.HI_PROB>=%s) 
                    OR (a.ML_TYPE='ML5630' AND a.HI_PROB>=%s AND a.STR_OU_HKJC_HDC=2.5) 
                    OR (a.ML_TYPE='LOGREG' AND a.HI_PROB>=%s) 
                    OR (a.ML_TYPE='TWIST' AND a.HI_PROB>=%s)
                    OR (a.ML_TYPE='TWIST2' AND a.HI_PROB>=%s)
                    -- OR (a.ML_TYPE='HKJC' AND a.HI_PROB>=%s)
                )
            ORDER BY 
                a.MATCH_DATETIME DESC,a.MATCH_ID,a.ML_TYPE
            """ % (hi_prob_35, hi_prob_253, hi_prob_5630, hi_prob_logreg, hi_prob_twist, hi_prob_twist2, hi_prob_hkjc)
    c.execute(sql)
    result = c.fetchall()
    for row in result:
        ml_type = str(row[0])
        match_id = int(row[1])
        match_datetime = str(row[2])
        league = str(row[3])
        home_team = str(row[4])
        away_team = str(row[5])
        home_ft_goal = row[6]
        if home_ft_goal == None:
            home_ft_goal = None
            away_ft_goal = None
        else:
            home_ft_goal = int(row[6])
            away_ft_goal = int(row[7])
        str_ou_macau_hdc = float(row[8])
        str_ou_macau_hi = float(row[9])
        str_ou_macau_lo = float(row[10])
        str_ou_hkjc_hdc = float(row[11])
        str_ou_hkjc_hi = float(row[12])
        str_ou_hkjc_lo = float(row[13])
        hi_prob = float(row[14])
        match_date = int(row[15])
        data_row.append([ml_type, match_id, match_datetime, league, home_team, away_team, home_ft_goal, away_ft_goal, str_ou_macau_hdc,
                         str_ou_macau_hi, str_ou_macau_lo, str_ou_hkjc_hdc, str_ou_hkjc_hi, str_ou_hkjc_lo, hi_prob, match_date])

    sheet.append_rows(data_row)
    exec_time = strftime('%Y-%m-%d %H:%M:%S')
    print('Last update time: %s' % exec_time)
    sheet.update_cell(1, 2, exec_time)

except cx_Oracle.Error as error:
    print('Oracle error - %s\n' % (error))
