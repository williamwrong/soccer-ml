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
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "overunderml.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ML Asian").worksheet("讓球ML")

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

    asian_25_und_prob = 0.57
    asian_75_fav_prob = 0.61
    asian_75_und_prob = 0.62
    twist_und_prob = 0.57
    twist2_und_prob = 0.54
    jctwist_und_prob = 0.53
    hda_und_prob = 0.63

    sql = """
            SELECT 
                a.ML_TYPE, a.MATCH_ID, a.MATCH_DATETIME, a.LEAGUE, a.HOME_TEAM, a.AWAY_TEAM, b.HOME_FT_GOAL, b.AWAY_FT_GOAL, a.STR_A_MACAU_HDC, a.STR_A_MACAU_H, a.STR_A_MACAU_A, a.STR_HDA_MACAU_H, a.STR_HDA_MACAU_D, a.STR_HDA_MACAU_A, a.STR_A_HKJC_HDC, a.STR_A_HKJC_H, a.STR_A_HKJC_A, a.STR_HDA_HKJC_H, a.STR_HDA_HKJC_D, a.STR_HDA_HKJC_A, a.FAV_PROB, a.UND_PROB,
                CASE WHEN a.ML_TYPE='ASIAN075' AND a.FAV_PROB>=%s THEN '上盤' ELSE '下盤' END AS PREDICTION, a.MATCH_DATE 
            FROM 
                ML_PREDICT_ASIAN a, MATCH_INFO b
            WHERE 
                a.MATCH_ID=b.MATCH_ID
                AND a.STR_A_HKJC_HDC IS NOT NULL
                AND a.STR_HDA_MACAU_H<a.STR_HDA_MACAU_A
                AND (a.STR_A_MACAU_HDC>0 OR (a.STR_A_HKJC_HDC=0 AND a.STR_A_MACAU_A<a.STR_A_MACAU_H))
                AND (
                    (a.ML_TYPE='ASIAN025' AND a.UND_PROB>=%s) 
                    OR (a.ML_TYPE='ASIAN075' AND a.FAV_PROB>=%s) 
                    OR (a.ML_TYPE='ASIAN075' AND a.UND_PROB>=%s) 
                    OR (a.ML_TYPE='TWIST' AND a.UND_PROB>=%s) 
                    OR (a.ML_TYPE='TWIST2' AND a.UND_PROB>=%s)
                    OR (a.ML_TYPE='JCTWIST' AND a.UND_PROB>=%s)
                    OR (a.ML_TYPE='HDA' AND a.UND_PROB>=%s)
                    ) 
            ORDER BY 
                a.MATCH_DATETIME DESC, a.MATCH_ID, a.ML_TYPE
            """ % (asian_75_fav_prob, asian_25_und_prob, asian_75_fav_prob, asian_75_und_prob, twist_und_prob, twist2_und_prob, jctwist_und_prob, hda_und_prob)
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
        # if math.isnan(home_ft_goal):
        if home_ft_goal == None:
            home_ft_goal = None
            away_ft_goal = None
        else:
            home_ft_goal = int(row[6])
            away_ft_goal = int(row[7])
        str_a_macau_hdc = float(row[8])
        str_a_macau_h = float(row[9])
        str_a_macau_a = float(row[10])
        str_hda_macau_h = float(row[11])
        str_hda_macau_d = float(row[12])
        str_hda_macau_a = float(row[13])
        str_a_hkjc_hdc = float(row[14])
        str_a_hkjc_h = float(row[15])
        str_a_hkjc_a = float(row[16])
        str_hda_hkjc_h = float(row[17])
        str_hda_hkjc_d = float(row[18])
        str_hda_hkjc_a = float(row[19])
        fav_prob = float(row[20])
        und_prob = float(row[21])
        pred = str(row[22])
        match_date = int(row[23])

        data_row.append([ml_type, match_id, match_datetime, league, home_team, away_team, home_ft_goal, away_ft_goal, str_a_macau_hdc, str_a_macau_h, str_a_macau_a, str_hda_macau_h, str_hda_macau_d,
                         str_hda_macau_a, str_a_hkjc_hdc, str_a_hkjc_h, str_a_hkjc_a, str_hda_hkjc_h, str_hda_hkjc_d, str_hda_hkjc_a, fav_prob, und_prob, pred, match_date])

    sheet.append_rows(data_row)
    exec_time = strftime('%Y-%m-%d %H:%M:%S')
    print('Last update time: %s' % exec_time)
    sheet.update_cell(1, 2, exec_time)

except cx_Oracle.Error as error:
    print('Oracle error - %s\n' % (error))
