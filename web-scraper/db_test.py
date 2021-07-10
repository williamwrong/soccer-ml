# from datetime import datetime
from datetime import datetime, time, timedelta
# today = datetime.now()
# previous_day = today - timedelta(days=1)
# previous_day = previous_day.strftime("%Y%m%d")
# print(today)
# print(previous_day)

print(datetime.strptime("2013-1-25", '%Y-%m-%d').strftime('%Y-%m-%d 00:00:00'))

# def is_midnight():
#     current_time = datetime.now().time
#     match_date = datetime.now()
#     # current_time = datetime.now()
#     if current_time >= time(0,0) and current_time <= time(4, 30):
#         print(match_date)
#         return match_date.strftime("%Y%m%d")
#     else:
#         match_date = current_time - timedelta(days=1)
#         return match_date
    
# print(is_midnight())

# def is_midnight():
#     current_time = datetime.now().time()
#     # today = datetime.now()
#     previous_day = current_time - timedelta(days=1)
#     previous_day = previous_day.strftime("%Y%m%d")
#     if current_time >= time(0,0) and current_time <= time(4, 30):
#         print('yesy')
    # if time(0,0) < time(4, 30):
        # return current_time >= time(0,0) and current_time <= time(4, 30)
    # else: # crosses midnight
    #     return current_time >= time(0,0) or current_time <= time(4, 30)

# print(is_midnight())


# match_date = datetime.now()
# current_time = datetime.now().time()
# if current_time >= time(0,0) and current_time <= time(4, 30):
#     previous_day = match_date - timedelta(days=1)
#     match_date = previous_day.strftime("%Y%m%d")
    
# print(match_date)

# match_date = datetime.now().strftime("%Y-%m-%d")
# # datetime.timedelta(days=1)
# hour = datetime.now().strftime("%H")
# if int(hour) < 4:
#     print('Previous day: %s' % (datetime.timedelta(days=1)))
# else:
#     print(match_date)
    
# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# import cx_Oracle

# connection = None
# try:
#     connection = cx_Oracle.connect(
#         'JW',
#         '901203',
#         'HOME-PC/XE',
#         encoding='UTF-8')

#     # show the version of the Oracle Database
#     print(connection.version)
    
#     c = connection.cursor()
#     c.execute('SELECT MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,HOME_FT_GOAL,AWAY_FT_GOAL,CASE WHEN HOME_FT_GOAL > AWAY_FT_GOAL THEN \'H\' WHEN HOME_FT_GOAL = AWAY_FT_GOAL THEN \'D\' ELSE \'A\' END HDA_RESULT,DRAW_MEAN,DRAW_MEDIAN,O_MACAU_D,O_BET365_D,O_HKJC_D,AWAY_MEAN,AWAY_MEDIAN,O_MACAU_A,O_BET365_A,O_HKJC_A,HKJC_ASIAN_HANDICAP,HKJC_ASIAN_AWAY FROM HDA_MEAN_VIEW WHERE MATCH_HANDICAP=\'上盤\' AND HOME_IND=0 AND DRAW_IND=1 AND AWAY_IND=1 ORDER BY MATCH_DATETIME DESC')
#     for row in c:
#         print(row[0],',',row[1])
# except cx_Oracle.Error as error:
#     print(error)
# finally:
#     # release the connection
#     if connection:
#         connection.close()
