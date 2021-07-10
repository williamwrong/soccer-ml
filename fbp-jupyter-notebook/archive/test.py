import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
from time import strftime

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("overunderml.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ML OverUnder").sheet1
data = sheet.get_all_records()

num_rows = sheet.row_count
pprint(data)
print(num_rows)

exec_time = strftime('%Y-%m-%d %H:%M:%S')
sheet.update_cell(1,2,exec_time)
print('Last update time: %s' % exec_time)

# gc = gspread.service_account(filename='overunderml.json')
# sheet = gc.open_by_key('1r9Qp88F7j5qIAf4-jxj7e6_8e48rVSaQ9ZWuYjF6uwI')
# worksheet = sheet.sheet1

# res = worksheet.row_values(3)
# print(res)
# print(num_rows)