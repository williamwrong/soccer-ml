#!/usr/bin/python
# coding:utf-8

from bs4 import BeautifulSoup
# from time import strftime
# from datetime import timedelta, date
from urllib.request import urlopen
from requests_html import HTMLSession
# import requests
# import codecs
# import sys
# import os



match_date = '20200803'
def scrape_by_date(match_date):
    print('Scrape mode: Single match day')
    print('Match date: [%s]' % match_date)
    
    url = 'http://bf.win007.com/football/big/Over_%s.htm' % match_date

    session = HTMLSession()

    r = session.get(url)
    r.html.render()
    r.html.encoding = 'utf-8'

    # Define HKJC leagues
    hkjc_league_list = '亞冠盃,俄盃,俄超,南球盃,墨西哥盃,墨西聯,墨西聯 春,巴甲,巴聖錦標,巴西盃,德乙,德甲,意甲,挪超,日職乙,日職聯,日超杯,智利甲,歐冠盃,歐霸盃,比利時盃,比甲,法乙,法甲,澳洲甲,球會友誼,瑞典盃,瑞典超,美冠盃,美職業,自由盃,英冠,英甲,英聯盃,英超,英足總盃,英錦賽,荷乙,荷甲,葡盃,葡超,蘇總盃,蘇超,西甲,阿根廷盃,阿甲,韓K聯'

    soup = BeautifulSoup(r.html.html, 'html.parser')
    live_table = soup.find('table', id = 'table_live')
    match_ids = []

    for row in live_table.find_all('tr'):
        try:
            style = row['style']
        except KeyError:
            try:
                league = row.find_all('td')[0].text
                league = ' '.join(league.split())
                if hkjc_league_list.find(league) != -1:
                    match_id = row.find_all('td')[4]['onclick'].split('(')[1].split(')')[0]
                    match_ids.append(match_id)
            except KeyError:
                pass

    total_matches = len(match_ids)
    print('[%s] %s match ids collected' % (match_date, str(total_matches)))

    # for match in match_ids:
    for match in range(total_matches):
        print('[%s] scraping [%s/%s]...' % (match_date, str(match+1), str(total_matches)))
        # scrape(match_ids[match])

scrape_by_date(match_date)
# url = 'http://op1.win007.com/oddslist/%s_2.htm' % match_date

# session = HTMLSession()

# r = session.get(url)
# r.html.render()
# r.html.encoding = 'utf-8'

# print(r.html.html)

# processed_bookmaker = 0
# bookmaker_list = '澳门,Crown,bet365(英国),易胜博(安提瓜和巴布达),伟德(直布罗陀),明陞(菲律宾),10BET(英国),金宝博(马恩岛),12BET(菲律宾),利记sbobet(英国),盈禾(菲律宾),18Bet,Pinnacle(荷兰),香港马会(中国香港)'

exit

