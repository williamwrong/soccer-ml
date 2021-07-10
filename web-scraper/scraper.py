#!/usr/bin/python
# coding:utf-8

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from time import strftime
from datetime import timedelta, datetime, date, time
from urllib.request import urlopen
from requests_html import HTMLSession
import socket
import requests
import codecs
import sys
import os
import cx_Oracle
import csv


def logger(severity, msg):
    print('[%s] -%s - %s' % (strftime('%Y-%m-%d %H:%M:%S'), severity, msg))


def get_match_info(match_date, mode):
    logger('I', 'Match date: %s' % match_date)

    logger('I', 'Start getting details from MATCHDAY_STATUS...')
    sql = 'SELECT COUNT(1) FROM MATCHDAY_STATUS WHERE MATCH_DATE=:match_date'
    c.execute(sql, [match_date])

    count = c.fetchone()[0]
    if count == 0:
        logger('I', 'Start collecting match info...')
        data_to_insert = []
        match_id_list = []
        session = HTMLSession()

        # Past data
        if mode == 'history':
            url = 'http://bf.win007.com/football/big/Over_%s.htm' % match_date

            r = session.get(url)
            r.html.render(timeout=timeout)
            r.html.encoding = 'utf-8'

            soup = BeautifulSoup(r.html.html, 'html.parser')
            live_table = soup.find('table', id = 'table_live')

            for row in live_table.find_all('tr'):
                try:
                    style = row['style']
                except KeyError:
                    try:
                        league = row.find_all('td')[0].text
                        league = ' '.join(league.split())
                        if hkjc_league_list.find(league) != -1:
                            match_id = row.find_all('td')[4]['onclick'].split('(')[1].split(')')[0]
                            home_team = row.find_all('td')[3].text.split(']')[1]
                            away_team = row.find_all('td')[5].text.split('[')[0]
                            home_ft_goal = None
                            away_ft_goal = None
                            home_ht_goal = None
                            away_ht_goal = None
                            match_id_list.append([match_id])
                            data_to_insert.append([match_id, match_date, league, home_team, away_team, home_ft_goal, away_ft_goal, home_ht_goal, away_ht_goal, match_date])
                    except KeyError:
                        pass
                    except IndexError:
                        pass

        # Current date
        if mode == 'current':
            print('Mode: %s' % mode)
            url = 'http://live.win007.com/indexall_big.aspx'

            r = session.get(url)
            r.html.render(timeout=timeout)
            r.html.encoding = 'utf-8'

            soup = BeautifulSoup(r.html.html, 'html.parser')
            live_table = soup.find('table', id = 'table_live')

            for row in live_table.find_all('tr'):
                try:
                    style = row['style']
                    if style.find('display:none') == -1:
                        try:
                            league = row.find_all('td')[1].text
                            league = ' '.join(league.split())
                            if hkjc_league_list.find(league) != -1:
                                match_id = row.find_all('td')[5]['onclick'].split('(')[1].split(')')[0]
                                home_team = row.find_all('td')[4].text
                                away_team = row.find_all('td')[6].text
                                status = row.find_all('td')[3].text
                                home_ft_goal = None
                                away_ft_goal = None
                                home_ht_goal = None
                                away_ht_goal = None
                                if status == '完':
                                    home_ft_goal = row.find_all('td')[5].text.split('-')[0]
                                    away_ft_goal = row.find_all('td')[5].text.split('-')[1]
                                    home_ht_goal = row.find_all('td')[7].find_all('span')[1].text.split('-')[0]
                                    away_ht_goal = row.find_all('td')[7].find_all('span')[1].text.split('-')[1]
                                match_id_list.append([match_id])
                                data_to_insert.append([match_id, match_date, league, home_team, away_team, home_ft_goal, away_ft_goal, home_ht_goal, away_ht_goal, match_date])
                                
                        except KeyError:
                            pass
                        except IndexError:
                            pass
                        
                except KeyError:
                    pass

        # Insert MATCH_INFO
        sql = 'DELETE FROM MATCH_INFO WHERE MATCH_ID=:match_id'
        c.executemany(sql, match_id_list)
        
        sql = 'INSERT INTO MATCH_INFO (MATCH_ID, MATCH_DATETIME, LEAGUE, HOME_TEAM, AWAY_TEAM, HOME_FT_GOAL, AWAY_FT_GOAL, HOME_HT_GOAL, AWAY_HT_GOAL, DATA_DATE) VALUES (:match_id, TO_DATE(:match_date,\'YYYY-MM-DD HH24:MI:SS\'), :league, :home_team, :away_team, :home_ft_goal, :away_ft_goal, :home_ht_goal, :away_ht_goal, :data_date)'
        c.executemany(sql, data_to_insert)

        # Insert MATCHDAY_STATUS
        sql = 'DELETE FROM MATCHDAY_STATUS WHERE MATCH_DATE=:match_date'
        c.execute(sql, [match_date])
        sql = 'INSERT INTO MATCHDAY_STATUS (MATCH_DATE, IS_SCRAPED) VALUES (:match_date, \'Y\')'
        c.execute(sql, [match_date])

        total_matches = len(data_to_insert)
        logger('I', '%s matches collected' % (str(total_matches)))
        
        connection.commit()
    
    else:
        logger('I', 'Match info already captured in database')


def recent_stat(match_id):
    logger('I', 'Start scraping recent stat... [%s]' % (match_id))
    url = 'http://zq.win007.com/analysis/%s.htm' % match_id
    
    session = HTMLSession()
    
    try:
        # r = session.get(url)
        # r.html.render(timeout=timeout)
        # r.html.encoding = 'utf-8'

        # soup = BeautifulSoup(r.html.html, 'html.parser')
        # recent_table = soup.find('table', id='table_com')
        # print(recent_table)
        
        browser.get(url)
        
        element = WebDriverWait(browser, 30).until(
            EC.presence_of_element_located((By.ID, "tr_com_h"))
        )
        
        home_total_gf = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[2]").text
        home_total_ga = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[3]").text
        home_net_gf = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[4]").text
        home_avg_gf = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[5]").text
        home_win_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_h']/td[6]").text.split('%')[0]) / 100,4))
        home_draw_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_h']/td[7]").text.split('%')[0]) / 100,4))
        home_lose_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_h']/td[8]").text.split('%')[0]) / 100,4))
        home_home_gf = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[9]").text
        home_home_ga = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[10]").text
        home_home_net_gf = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[11]").text
        home_home_avg_gf = browser.find_element_by_xpath("//*[@id='tr_com_h']/td[12]").text
        home_home_win_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_h']/td[13]").text.split('%')[0]) / 100,4))
        home_home_draw_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_h']/td[14]").text.split('%')[0]) / 100,4))
        home_home_lose_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_h']/td[15]").text.split('%')[0]) / 100,4))
        
        away_total_gf = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[2]").text
        away_total_ga = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[3]").text
        away_net_gf = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[4]").text
        away_avg_gf = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[5]").text
        away_win_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_a']/td[6]").text.split('%')[0]) / 100,4))
        away_draw_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_a']/td[7]").text.split('%')[0]) / 100,4))
        away_lose_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_a']/td[8]").text.split('%')[0]) / 100,4))
        away_away_gf = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[9]").text
        away_away_ga = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[10]").text
        away_away_net_gf = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[11]").text
        away_away_avg_gf = browser.find_element_by_xpath("//*[@id='tr_com_a']/td[12]").text
        away_away_win_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_a']/td[13]").text.split('%')[0]) / 100,4))
        away_away_draw_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_a']/td[14]").text.split('%')[0]) / 100,4))
        away_away_lose_rate = str(round(float(browser.find_element_by_xpath("//*[@id='tr_com_a']/td[15]").text.split('%')[0]) / 100,4))
        
        # print(home_total_gf,home_total_ga,home_net_gf,home_avg_gf,home_win_rate,home_draw_rate,home_lose_rate,home_home_gf,home_home_ga,home_home_net_gf,home_home_avg_gf,home_home_win_rate,home_home_draw_rate,home_home_lose_rate)
        # print(away_total_gf,away_total_ga,away_net_gf,away_avg_gf,away_win_rate,away_draw_rate,away_lose_rate,away_away_gf,away_away_ga,away_away_net_gf,away_away_avg_gf,away_away_win_rate,away_away_draw_rate,away_away_lose_rate)

        # Insert RECENT_RAW
        sql = 'DELETE FROM RECENT_RAW WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        sql = 'INSERT INTO RECENT_RAW (MATCH_ID, HOME_TOTAL_GF, HOME_TOTAL_GA, HOME_NET_GF, HOME_AVG_GF, HOME_WIN_RATE, HOME_DRAW_RATE, HOME_LOSE_RATE, HOME_HOME_GF, HOME_HOME_GA, HOME_HOME_NET_GF, HOME_HOME_AVG_GF, HOME_HOME_WIN_RATE, HOME_HOME_DRAW_RATE, HOME_HOME_LOSE_RATE, AWAY_TOTAL_GF, AWAY_TOTAL_GA, AWAY_NET_GF, AWAY_AVG_GF, AWAY_WIN_RATE, AWAY_DRAW_RATE, AWAY_LOSE_RATE, AWAY_AWAY_GF, AWAY_AWAY_GA, AWAY_AWAY_NET_GF, AWAY_AWAY_AVG_GF, AWAY_AWAY_WIN_RATE, AWAY_AWAY_DRAW_RATE, AWAY_AWAY_LOSE_RATE) VALUES (:match_id, :home_total_gf, :home_total_ga, :home_net_gf, :home_avg_gf, :home_win_rate, :home_draw_rate, :home_lose_rate, :home_home_gf, :home_home_ga, :home_home_net_gf, :home_home_avg_gf, :home_home_win_rate, :home_home_draw_rate, :home_home_lose_rate, :away_total_gf, :away_total_ga, :away_net_gf, :away_avg_gf, :away_win_rate, :away_draw_rate, :away_lose_rate, :away_away_gf, :away_away_ga, :away_away_net_gf, :away_away_avg_gf, :away_away_win_rate, :away_away_draw_rate, :away_away_lose_rate)'
        c.execute(sql, [match_id,home_total_gf,home_total_ga,home_net_gf,home_avg_gf,home_win_rate,home_draw_rate,home_lose_rate,home_home_gf,home_home_ga,home_home_net_gf,home_home_avg_gf,home_home_win_rate,home_home_draw_rate,home_home_lose_rate,away_total_gf,away_total_ga,away_net_gf,away_avg_gf,away_win_rate,away_draw_rate,away_lose_rate,away_away_gf,away_away_ga,away_away_net_gf,away_away_avg_gf,away_away_win_rate,away_away_draw_rate,away_away_lose_rate])
        connection.commit()

    except ValueError as ve:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] recent_stat() passed - [%s] %s\n' % (match_id, ve, url))
        file.close()
        pass
    except AttributeError as ae:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] recent_stat() passed - [%s] %s\n' % (match_id, ae, url))
        file.close()
        pass
    except IndexError as ie:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] recent_stat() passed - [%s] %s\n' % (match_id, ie, url))
        file.close()
        pass
    except TimeoutError as toe:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] recent_stat() passed - [%s] %s\n' % (match_id, toe, url))
        file.close()
        pass


def hilo(match_id):
    logger('I', 'Start scraping hilo... [%s]' % (match_id))
    url = 'http://vip.win007.com/OverDown_n.aspx?id=%s&l=1' % match_id
    data_to_insert = []

    try:
        r = urlopen(url)
        soup = BeautifulSoup(r, 'html.parser', from_encoding='gb18030')

        # Create/open file
        # file = codecs.open(hilo_file, "a+", "utf-8")

        # Odd table
        odd_table = soup.find_all('table', class_='font13')[0]
        curr_bookmaker = ''
        prev_bookmaker = ''
        is_hkjc_exist = 0

        for iteration, rows in enumerate(odd_table.find_all('tr')):
            try:
                if iteration > 1:
                    curr_bookmaker = rows.find_all('td')[0].text
                    if curr_bookmaker == '香港马会':
                        is_hkjc_exist = 1
                    if curr_bookmaker == '':
                        curr_bookmaker = prev_bookmaker
                    else:
                        prev_bookmaker = curr_bookmaker
                    line = rows.find_all('td')[1].text
                    if line == '':
                        line = '盘口1'
                    start_hi = rows.find_all('td')[2].text
                    start_handicap = rows.find_all('td')[3]['goals']
                    start_lo = rows.find_all('td')[4].text
                    end_hi = rows.find_all('td')[8].text
                    end_handicap = rows.find_all('td')[9]['goals']
                    end_lo = rows.find_all('td')[10].text

                    odd_data = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (match_id, curr_bookmaker, line, start_hi, start_handicap, start_lo, end_hi, end_handicap, end_lo)
                    # file.write(odd_data)
                    
                    # Prepare data for insertion
                    data_to_insert.append([match_id, curr_bookmaker, line, start_hi, start_handicap, start_lo, end_hi, end_handicap, end_lo])
            except IndexError:
                logger('E', '[%s] IndexError. pass' % match_id)
                # sql = 'DELETE FROM MATCH_INFO WHERE MATCH_ID=:match_id'
                # c.execute(sql, [match_id])
                pass
            except KeyError:
                # logger('E', '[%s] KeyError. pass' % match_id)
                # sql = 'DELETE FROM MATCH_INFO WHERE MATCH_ID=:match_id'
                # c.execute(sql, [match_id])
                pass

        # file.close()
        
        # Skip if HKJC not exist
        if is_hkjc_exist == 0:
            logger('I', '[%s] HKJC not exist. Delete from match info' % match_id)
            # sql = 'DELETE FROM MATCH_INFO WHERE MATCH_ID=:match_id'
            # c.execute(sql, [match_id])
            # connection.commit()
            return False
        
        # Match info
        match_info = soup.find('div', class_='vs').find_all('div')[0].text
        match_info = ' '.join(match_info.split())
        match_datetime = match_info.split(' ')[1] + ' ' + match_info.split(' ')[2].split('\xa0')[0] + ':00'
        league = match_info.split(' ')[0].strip()
        home_team = soup.find('div', class_='home').find('a').text.split(' ')[0]
        away_team = soup.find('div', class_='guest').find('a').text.split(' ')[0]
        try:
            ht_goal = soup.find_all('div', class_='end')[0].find_all('div')[1].find_all('span')[1].text
            home_ft_goal = soup.find_all('div', class_='score')[0].text.strip()
            away_ft_goal = soup.find_all('div', class_='score')[1].text.strip()
            home_ht_goal = ht_goal.split('-')[0].split('(')[1]
            away_ht_goal = ht_goal.split('-')[1].split(')')[0]
            
        except IndexError:
            try:
                home_ft_goal = ''
                away_ft_goal = ''
                home_ht_goal = soup.find_all('div', class_='score')[0].text.strip()
                away_ht_goal = soup.find_all('div', class_='score')[1].text.strip()
            except IndexError:
                home_ft_goal = ''
                away_ft_goal = ''
                home_ht_goal = ''
                away_ht_goal = ''
        
        # Update MATCH_INFO
        sql = 'UPDATE MATCH_INFO SET MATCH_DATETIME=TO_DATE(:match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),HOME_FT_GOAL=:home_ft_goal,AWAY_FT_GOAL=:away_ft_goal,HOME_HT_GOAL=:home_ht_goal,AWAY_HT_GOAL=:away_ht_goal WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_datetime, home_ft_goal, away_ft_goal, home_ht_goal, away_ht_goal, match_id])
        if c.rowcount == 0:
            sql = 'INSERT INTO MATCH_INFO (MATCH_ID,MATCH_DATETIME,LEAGUE,HOME_TEAM,AWAY_TEAM,HOME_FT_GOAL,AWAY_FT_GOAL,HOME_HT_GOAL,AWAY_HT_GOAL,DATA_DATE) VALUES (:match_id,TO_DATE(:match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),:league,:home_team,:away_team,:home_ft_goal,:away_ft_goal,:home_ht_goal,:away_ht_goal,TO_CHAR(TO_DATE(:match_datetime,\'YYYY-MM-DD HH24:MI:SS\'),\'YYYYMMDD\'))'
            c.execute(sql, [match_id,match_datetime,league,home_team,away_team,home_ft_goal,away_ft_goal,home_ht_goal,away_ht_goal,match_datetime])
        
        # Insert HILO_RAW
        sql = 'DELETE FROM HILO_RAW WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        sql = 'INSERT INTO HILO_RAW (MATCH_ID,BOOKMAKER,LINE,START_HI,START_HANDICAP,START_LO,END_HI,END_HANDICAP,END_LO) VALUES (:match_id,:curr_bookmaker,:line,:start_hi,:start_handicap,:start_lo,:end_hi,:end_handicap,:end_lo)'
        c.executemany(sql, data_to_insert)
        connection.commit()
        
        return True
        
    except AttributeError as ae:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] hilo() passed - [%s] %s\n' % (match_id, ae, url))
        file.close()
        pass
    except IndexError as ie:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] hilo() passed - [%s] %s\n' % (match_id, ie, url))
        file.close()
        pass
    except TimeoutError as toe:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] hilo() passed - [%s] %s\n' % (match_id, toe, url))
        file.close()
        pass


def asian(match_id):
    logger('I', 'Start scraping asian... [%s]' % (match_id))
    url = 'http://vip.win007.com/AsianOdds_n.aspx?id=%s&l=1' % match_id
    data_to_insert = []

    try:
        r = urlopen(url)
        soup = BeautifulSoup(r, 'html.parser', from_encoding='gb18030')

        # Create/open file
        # file = codecs.open(asian_file, "a+", "utf-8")

        # Odd table
        odd_table = soup.find_all('table', class_='font13')[0].find_all('tr')
        curr_bookmaker = ''
        prev_bookmaker = ''

        for row in range(len(odd_table)):
            try:
                if row > 1:
                    curr_bookmaker = odd_table[row].find_all('td')[0].text
                    if curr_bookmaker == '':
                        curr_bookmaker = prev_bookmaker
                    else:
                        prev_bookmaker = curr_bookmaker
                    line = odd_table[row].find_all('td')[1].text
                    if line == '':
                        line = '盘口1'
                    start_home = odd_table[row].find_all('td')[2].text
                    start_handicap = odd_table[row].find_all('td')[3]['goals']
                    start_away = odd_table[row].find_all('td')[4].text
                    end_home = odd_table[row].find_all('td')[8].text
                    end_handicap = odd_table[row].find_all('td')[9]['goals']
                    end_away = odd_table[row].find_all('td')[10].text

                    # print('bookmaker: %s' % curr_bookmaker)
                    # print('line: %s' % line)
                    # print('start_home: %s' % start_home)
                    # print('start_handicap: %s' % start_handicap)
                    # print('start_away: %s' % start_away)
                    # print('end_home: %s' % end_home)
                    # print('end_handicap: %s' % end_handicap)
                    # print('end_away: %s' % end_away)

                    odd_data = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (match_id, curr_bookmaker, line, start_home, start_handicap, start_away, end_home, end_handicap, end_away)
                    # file.write(odd_data)
                    
                    # Prepare data for insertion
                    data_to_insert.append([match_id, curr_bookmaker, line, start_home, start_handicap, start_away, end_home, end_handicap, end_away])

            except KeyError:
                pass

        # file.close()
        
        # Write db
        sql = 'DELETE FROM ASIAN_RAW WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        sql = 'INSERT INTO ASIAN_RAW (MATCH_ID,BOOKMAKER,LINE,START_HOME,START_HANDICAP,START_AWAY,END_HOME,END_HANDICAP,END_AWAY) VALUES (:match_id,:curr_bookmaker,:line,:start_home,:start_handicap,:start_away,:end_home,:end_handicap,:end_away)'
        c.executemany(sql, data_to_insert)
        connection.commit()
        
    except AttributeError as ae:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] asian() passed - [%s] %s\n' % (match_id, ae, url))
        file.close()
        pass
    except IndexError as ie:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] asian() passed - [%s] %s\n' % (match_id, ie, url))
        file.close()
        pass
    except TimeoutError as toe:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] asian() passed - [%s] %s\n' % (match_id, toe, url))
        file.close()
        pass


def hda(match_id):
    logger('I', 'Start scraping hda... [%s]' % match_id)
    url = 'http://op1.win007.com/oddslist/%s_2.htm' % match_id
    # is_success = 'Y'
    data_to_insert = []
    
    try:
        sql = 'DELETE FROM HDA_RAW WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        
        browser.get(url)
        
        for i in range(2):
            browser.find_element_by_xpath("//select[@name='sel_showType']/option[text()='初盘']").click()
            if i == 1:
                browser.find_element_by_xpath("//select[@name='sel_showType']/option[text()='即时盘']").click()
            rows = browser.find_elements_by_xpath("//*[@id='oddsList_tab']/tbody/tr")

            handicap_type = i
            processed_bookmaker = 0
            bookmaker_list = '澳门,Crown,bet365(英国),易胜博(安提瓜和巴布达),伟德(直布罗陀),明陞(菲律宾),10BET(英国),金宝博(马恩岛),12BET(菲律宾),利记sbobet(英国),盈禾(菲律宾),18Bet,Pinnacle(荷兰),香港马会(中国香港)'

            # Create/open file
            # file = codecs.open(hda_file, 'a+', 'utf-8')

            for row in range(len(rows)):
                if row < 40:
                    odds = rows[row].text.replace(' ',',').replace('bet,365','bet365').replace('5,Dimes(哥斯达黎加)','5Dimes(哥斯达黎加)')
                    bookmaker = odds.split(',')[0]

                    if processed_bookmaker == 14:
                        break
                    else:
                        if bookmaker_list.find(bookmaker) != -1:
                            bookmaker = bookmaker.split('(')[0]
                            home_odd = odds.split(',')[1]
                            draw_odd = odds.split(',')[2]
                            away_odd = odds.split(',')[3]
                            home_win_rate = str(round(float(odds.split(',')[4]) / 100,4))
                            draw_win_rate = str(round(float(odds.split(',')[5]) / 100,4))
                            away_win_rate = str(round(float(odds.split(',')[6]) / 100,4))
                            return_rate = str(round(float(odds.split(',')[7]) / 100,4))
                            home_kelly = odds.split(',')[8]
                            draw_kelly = odds.split(',')[9]
                            away_kelly = odds.split(',')[10]
                            processed_bookmaker += 1

                            bookmaker_odd = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (match_id,bookmaker,handicap_type,home_odd,draw_odd,away_odd,home_win_rate,draw_win_rate,away_win_rate,return_rate,home_kelly,draw_kelly,away_kelly)
                            # file.write(bookmaker_odd)

                            # Prepare data for insertion
                            data_to_insert.append([match_id, bookmaker, handicap_type, home_odd, draw_odd, away_odd, home_win_rate, draw_win_rate, away_win_rate, return_rate, home_kelly, draw_kelly, away_kelly])

        # file.close()
        
        # Write db
        sql = 'INSERT INTO HDA_RAW (MATCH_ID,BOOKMAKER,HANDICAP_TYPE,HOME_ODD,DRAW_ODD,AWAY_ODD,HOME_WIN_RATE,DRAW_WIN_RATE,AWAY_WIN_RATE,RETURN_RATE,HOME_KELLY,DRAW_KELLY,AWAY_KELLY) VALUES (:match_id,:bookmaker,:handicap_type,:home_odd,:draw_odd,:away_odd,:home_win_rate,:draw_win_rate,:away_win_rate,:return_rate,:home_kelly,:draw_kelly,:away_kelly)'
        c.executemany(sql, data_to_insert)
        
        # return is_success

    except AttributeError as ae:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] hda() passed - [%s] %s\n' % (match_id, ae, url))
        file.close()
        pass
    except IndexError as ie:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] hda() passed - [%s] %s\n' % (match_id, ie, url))
        file.close()
        pass
    except TimeoutError as toe:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] hda() passed - [%s] %s\n' % (match_id, toe, url))
        file.close()
        pass
    except TimeoutException as to:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] - [%s] %s\n' % (match_id, to, url))
        file.close()
        # is_success = 'N'
        # return is_success


def betfa(match_id):
    logger('I', 'Start scraping betfa... [%s]' % match_id)
    url = 'http://vip.win007.com/betfa/single.aspx?id=%s&l=1' % match_id
    
    try:
        r = urlopen(url)
        soup = BeautifulSoup(r, 'html.parser', from_encoding='gb18030')

        betfa_table = soup.find('table', class_='tcenter')
        # 百家平均
        all_house_home_idx = betfa_table.find_all('tr')[2].find_all('td')[1].text
        all_house_draw_idx = betfa_table.find_all('tr')[3].find_all('td')[1].text
        all_house_away_idx = betfa_table.find_all('tr')[4].find_all('td')[1].text
        all_house_home_win_rate = str(round(float(betfa_table.find_all('tr')[2].find_all('td')[2].text.split('%')[0]) / 100,4))
        all_house_draw_win_rate = str(round(float(betfa_table.find_all('tr')[3].find_all('td')[2].text.split('%')[0]) / 100,4))
        all_house_away_win_rate = str(round(float(betfa_table.find_all('tr')[4].find_all('td')[2].text.split('%')[0]) / 100,4))
        all_house_return_rate = str(round(float(betfa_table.find_all('tr')[2].find_all('td')[3].text.split('%')[0]) / 100,4))
        # 必发成交数据
        betfa_home_idx = betfa_table.find_all('tr')[2].find_all('td')[4].text
        betfa_draw_idx = betfa_table.find_all('tr')[3].find_all('td')[3].text
        betfa_away_idx = betfa_table.find_all('tr')[4].find_all('td')[3].text
        betfa_home_win_rate = str(round(float(betfa_table.find_all('tr')[2].find_all('td')[5].text.split('%')[0]) / 100,4))
        betfa_draw_win_rate = str(round(float(betfa_table.find_all('tr')[3].find_all('td')[4].text.split('%')[0]) / 100,4))
        betfa_away_win_rate = str(round(float(betfa_table.find_all('tr')[4].find_all('td')[4].text.split('%')[0]) / 100,4))
        betfa_return_rate = str(round(float(betfa_table.find_all('tr')[2].find_all('td')[6].text.split('%')[0]) / 100,4))
        betfa_home_volume = betfa_table.find_all('tr')[2].find_all('td')[7].text
        betfa_draw_volume = betfa_table.find_all('tr')[3].find_all('td')[5].text
        betfa_away_volume = betfa_table.find_all('tr')[4].find_all('td')[5].text
        betfa_home_trade_rate = str(round(float(betfa_table.find_all('tr')[2].find_all('td')[8].text.split('%')[0]) / 100,4))
        betfa_draw_trade_rate = str(round(float(betfa_table.find_all('tr')[3].find_all('td')[6].text.split('%')[0]) / 100,4))
        betfa_away_trade_rate = str(round(float(betfa_table.find_all('tr')[4].find_all('td')[6].text.split('%')[0]) / 100,4))
        # 必发转换亚盘
        betfa_home_handicap = betfa_table.find_all('tr')[2].find_all('td')[9].text
        betfa_handicap = betfa_table.find_all('tr')[3].find_all('td')[7].text
        betfa_away_handicap = betfa_table.find_all('tr')[4].find_all('td')[7].text
        # 庄家盈亏
        betfa_home_pl = betfa_table.find_all('tr')[2].find_all('td')[10].text
        betfa_draw_pl = betfa_table.find_all('tr')[3].find_all('td')[8].text
        betfa_away_pl = betfa_table.find_all('tr')[4].find_all('td')[8].text
        # 盈亏指数
        betfa_home_pl_idx = betfa_table.find_all('tr')[2].find_all('td')[11].text
        betfa_draw_pl_idx = betfa_table.find_all('tr')[3].find_all('td')[9].text
        betfa_away_pl_idx = betfa_table.find_all('tr')[4].find_all('td')[9].text
        # 冷热指数
        betfa_home_hc_idx = betfa_table.find_all('tr')[2].find_all('td')[12].text
        betfa_draw_hc_idx = betfa_table.find_all('tr')[3].find_all('td')[10].text
        betfa_away_hc_idx = betfa_table.find_all('tr')[4].find_all('td')[10].text
        # 购买倾向
        betfa_home_buyer = betfa_table.find_all('tr')[2].find_all('td')[13].text
        betfa_draw_buyer = betfa_table.find_all('tr')[3].find_all('td')[11].text
        betfa_away_buyer = betfa_table.find_all('tr')[4].find_all('td')[11].text
        # 出售倾向
        betfa_home_seller_idx = betfa_table.find_all('tr')[2].find_all('td')[14].text
        betfa_draw_seller_idx = betfa_table.find_all('tr')[3].find_all('td')[12].text
        betfa_away_seller_idx = betfa_table.find_all('tr')[4].find_all('td')[12].text
        # 买卖单汇总
        volume_table = soup.find_all('table')[6]
        total_home_buyer = volume_table.find_all('tr')[2].find_all('td')[2].text
        total_draw_buyer = volume_table.find_all('tr')[2].find_all('td')[4].text
        total_away_buyer = volume_table.find_all('tr')[2].find_all('td')[6].text
        total_home_seller = volume_table.find_all('tr')[3].find_all('td')[2].text
        total_draw_seller = volume_table.find_all('tr')[3].find_all('td')[4].text
        total_away_seller = volume_table.find_all('tr')[3].find_all('td')[6].text

        # Create/open file
        # file = codecs.open(betfa_file, "a+", "utf-8")
        data = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (match_id,all_house_home_idx,all_house_draw_idx,all_house_away_idx,all_house_home_win_rate,all_house_draw_win_rate,all_house_away_win_rate,all_house_return_rate,betfa_home_idx,betfa_draw_idx,betfa_away_idx,betfa_home_win_rate,betfa_draw_win_rate,betfa_away_win_rate,betfa_return_rate,betfa_home_volume,betfa_draw_volume,betfa_away_volume,betfa_home_trade_rate,betfa_draw_trade_rate,betfa_away_trade_rate,betfa_home_handicap,betfa_handicap,betfa_away_handicap,betfa_home_pl,betfa_draw_pl,betfa_away_pl,betfa_home_pl_idx,betfa_draw_pl_idx,betfa_away_pl_idx,betfa_home_hc_idx,betfa_draw_hc_idx,betfa_away_hc_idx,betfa_home_buyer,betfa_draw_buyer,betfa_away_buyer,betfa_home_seller_idx,betfa_draw_seller_idx,betfa_away_seller_idx,total_home_buyer,total_draw_buyer,total_away_buyer,total_home_seller,total_draw_seller,total_away_seller)
        # file.write(data)
        # file.close()
        
        if (betfa_home_win_rate == 'nan'):
            betfa_home_win_rate = 0
        if (betfa_draw_win_rate == 'nan'):
            betfa_draw_win_rate = 0
        if (betfa_away_win_rate == 'nan'):
            betfa_away_win_rate = 0
        
        # Write db
        sql = 'DELETE FROM BETFA_RAW WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        sql = 'INSERT INTO BETFA_RAW (MATCH_ID,ALL_HOUSE_HOME_IDX,ALL_HOUSE_DRAW_IDX,ALL_HOUSE_AWAY_IDX,ALL_HOUSE_HOME_WIN_RATE,ALL_HOUSE_DRAW_WIN_RATE,ALL_HOUSE_AWAY_WIN_RATE,ALL_HOUSE_RETURN_RATE,BETFA_HOME_IDX,BETFA_DRAW_IDX,BETFA_AWAY_IDX,BETFA_HOME_WIN_RATE,BETFA_DRAW_WIN_RATE,BETFA_AWAY_WIN_RATE,BETFA_RETURN_RATE,BETFA_HOME_VOLUME,BETFA_DRAW_VOLUME,BETFA_AWAY_VOLUME,BETFA_HOME_TRADE_RATE,BETFA_DRAW_TRADE_RATE,BETFA_AWAY_TRADE_RATE,BETFA_HOME_HANDICAP,BETFA_HANDICAP,BETFA_AWAY_HANDICAP,BETFA_HOME_PL,BETFA_DRAW_PL,BETFA_AWAY_PL,BETFA_HOME_PL_IDX,BETFA_DRAW_PL_IDX,BETFA_AWAY_PL_IDX,BETFA_HOME_HC_IDX,BETFA_DRAW_HC_IDX,BETFA_AWAY_HC_IDX,BETFA_HOME_BUYER,BETFA_DRAW_BUYER,BETFA_AWAY_BUYER,BETFA_HOME_SELLER_IDX,BETFA_DRAW_SELLER_IDX,BETFA_AWAY_SELLER_IDX,TOTAL_HOME_BUYER,TOTAL_DRAW_BUYER,TOTAL_AWAY_BUYER,TOTAL_HOME_SELLER,TOTAL_DRAW_SELLER,TOTAL_AWAY_SELLER) VALUES (:match_id,:all_house_home_idx,:all_house_draw_idx,:all_house_away_idx,:all_house_home_win_rate,:all_house_draw_win_rate,:all_house_away_win_rate,:all_house_return_rate,:betfa_home_idx,:betfa_draw_idx,:betfa_away_idx,:betfa_home_win_rate,:betfa_draw_win_rate,:betfa_away_win_rate,:betfa_return_rate,:betfa_home_volume,:betfa_draw_volume,:betfa_away_volume,:betfa_home_trade_rate,:betfa_draw_trade_rate,:betfa_away_trade_rate,:betfa_home_handicap,:betfa_handicap,:betfa_away_handicap,:betfa_home_pl,:betfa_draw_pl,:betfa_away_pl,:betfa_home_pl_idx,:betfa_draw_pl_idx,:betfa_away_pl_idx,:betfa_home_hc_idx,:betfa_draw_hc_idx,:betfa_away_hc_idx,:betfa_home_buyer,:betfa_draw_buyer,:betfa_away_buyer,:betfa_home_seller_idx,:betfa_draw_seller_idx,:betfa_away_seller_idx,:total_home_buyer,:total_draw_buyer,:total_away_buyer,:total_home_seller,:total_draw_seller,:total_away_seller)'
        c.execute(sql, [match_id,all_house_home_idx,all_house_draw_idx,all_house_away_idx,all_house_home_win_rate,all_house_draw_win_rate,all_house_away_win_rate,all_house_return_rate,betfa_home_idx,betfa_draw_idx,betfa_away_idx,betfa_home_win_rate,betfa_draw_win_rate,betfa_away_win_rate,betfa_return_rate,betfa_home_volume,betfa_draw_volume,betfa_away_volume,betfa_home_trade_rate,betfa_draw_trade_rate,betfa_away_trade_rate,betfa_home_handicap,betfa_handicap,betfa_away_handicap,betfa_home_pl,betfa_draw_pl,betfa_away_pl,betfa_home_pl_idx,betfa_draw_pl_idx,betfa_away_pl_idx,betfa_home_hc_idx,betfa_draw_hc_idx,betfa_away_hc_idx,betfa_home_buyer,betfa_draw_buyer,betfa_away_buyer,betfa_home_seller_idx,betfa_draw_seller_idx,betfa_away_seller_idx,total_home_buyer,total_draw_buyer,total_away_buyer,total_home_seller,total_draw_seller,total_away_seller])
        sql = 'UPDATE MATCH_INFO SET IS_COMPLETE=\'Y\' WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        connection.commit()
        
    except AttributeError as ae:
        # file = codecs.open(error_file, 'a+', 'utf-8')
        # file.write('[%s] betfa() passed - [%s] %s\n' % (match_id, ae, url))
        # file.close()
        sql = 'UPDATE MATCH_INFO SET IS_COMPLETE=\'Y\' WHERE MATCH_ID=:match_id'
        c.execute(sql, [match_id])
        connection.commit()
        pass
    except IndexError as ie:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] betfa() passed - [%s] %s\n' % (match_id, ie, url))
        file.close()
        pass
    except TimeoutError as toe:
        file = codecs.open(error_file, 'a+', 'utf-8')
        file.write('[%s] betfa() passed - [%s] %s\n' % (match_id, toe, url))
        file.close()
        pass


def scrape(mode, param):
    logger('I', 'Scrape mode: %s' % mode)
    logger('I', 'Param: %s' % param)
    cnt = 0
    
    if mode == 'id':
        if hilo(param):
            recent_stat(param)
            asian(param)
            hda(param)
            betfa(param)

    if mode == 'history':
        match_date = datetime.strptime(param, '%Y%m%d').strftime('%Y-%m-%d')
        sql = 'SELECT MATCH_ID FROM MATCH_INFO WHERE DATA_DATE=:param AND (IS_COMPLETE IS NULL OR HOME_FT_GOAL IS NULL)'
        c.execute(sql, [param])
        result = c.fetchall()
        for row in result:
            cnt = cnt + 1
            logger('I', '====================>> [%s] - [%s/%s] <<====================' % (match_date, str(cnt), str(len(result))))
            match_id = row[0]
            # hda(match_id)
            if hilo(match_id):
                recent_stat(match_id)
                asian(match_id)
                hda(match_id)
                betfa(match_id)
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@EDITED@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # sql = 'UPDATE MATCH_INFO SET IS_COMPLETE=\'Y\' WHERE MATCH_ID=:match_id'
                # c.execute(sql, [match_id])
                # connection.commit()
                
        # Insert MATCHDAY_STATUS
        sql = 'SELECT COUNT(1) FROM MATCH_INFO WHERE IS_COMPLETE IS NULL AND DATA_DATE=:param'
        c.execute(sql, [param])
        count = c.fetchone()[0]
        if count == 0:
            sql = 'UPDATE MATCHDAY_STATUS SET IS_SCRAPED=\'Y\' WHERE MATCH_DATE=:param'
            c.execute(sql, [param])
            logger('I', '[%s] MATCHDAY_STATUS marked \'Y\'' % param)
            connection.commit()

    if mode == 'current':
        match_date = datetime.strptime(param, '%Y%m%d').strftime('%Y-%m-%d')
        sql = 'SELECT MATCH_ID FROM MATCH_INFO WHERE MATCH_DATETIME >= TO_DATE(:match_date, \'YYYY-MM-DD\') AND HOME_FT_GOAL IS NULL AND DATA_DATE=:param'
        c.execute(sql, [match_date, param])
        result = c.fetchall()
        for row in result:
            cnt = cnt + 1
            logger('I', '====================>> [%s] - [%s/%s] <<====================' % (match_date, str(cnt), str(len(result))))
            match_id = row[0]
            if hilo(match_id):
                recent_stat(match_id)
                asian(match_id)
                hda(match_id)
                betfa(match_id)

    if mode == 'upcoming':
        sql = 'SELECT MATCH_ID FROM MATCH_INFO WHERE DATA_DATE=:param AND HOME_FT_GOAL IS NULL AND MATCH_DATETIME < SYSDATE + INTERVAL \'1\' HOUR'
        c.execute(sql, [param])
        result = c.fetchall()
        for row in result:
            cnt = cnt + 1
            logger('I', '====================>> [%s/%s] <<====================' % (str(cnt), str(len(result))))
            match_id = row[0]
            if hilo(match_id):
                recent_stat(match_id)
                asian(match_id)
                hda(match_id)
                betfa(match_id)


# Main
curr_path = os.getcwd()
db_user = 'JW'
db_password = '901203'
db_dsn = 'HOME-PC/XE'
db_encoding = 'UTF-8'

# Current match date
current_match_date = datetime.now()
current_time = datetime.now().time()
if current_time >= time(0,0) and current_time <= time(11, 00):
    previous_day = current_match_date - timedelta(days=1)
    current_match_date = previous_day
current_match_date = current_match_date.strftime("%Y%m%d")

# Scrape mode (1=match_id, 2=date, 3=current_date, 4=upcoming, 5=history)
if len(sys.argv) > 2:
    mode = sys.argv[1]
    param = sys.argv[2]
elif len(sys.argv) == 2:
    mode = sys.argv[1]
    if mode == '1':
        # Define match id
        print('>>> Please enter match ID: e.g. 1872935')
        param = input()
    if mode == '2':
        # Define match dates
        print('>>> Please enter date: e.g. 20200801')
        param = input()
    if mode == '3' or mode == '4':
        param = current_match_date
    if mode == '5':
        param = 'history'
else:
    print('>>> Please enter mode: 1=match_id, 2=date, 3=current_date, 4=upcoming, 5=history')
    mode = input()
    if mode == '1':
        # Define match id
        print('>>> Please enter match ID: e.g. 1872935')
        param = input()
    if mode == '2':
        # Define match dates
        print('>>> Please enter date: e.g. 20200801')
        param = input()
    if mode == '3' or mode == '4':
        param = current_match_date
    if mode == '5':
        param = 'history'

# Define filename
asian_file = '%s\\data\\%s_asian.csv' % (curr_path, param)
hilo_file = '%s\\data\\%s_hilo.csv' % (curr_path, param)
hda_file = '%s\\data\\%s_hda.csv' % (curr_path, param)
betfa_file = '%s\\data\\%s_betfa.csv' % (curr_path, param)
match_info_file = '%s\\data\\%s_match_info.csv' % (curr_path, param)
error_file = '%s\\data\\%s_error.txt' % (curr_path, param)
hilo_asian_paths = [hilo_file, asian_file]
# print('error_file: %s' % error_file)

# Delete files
if os.path.exists(asian_file):
    os.remove(asian_file)
if os.path.exists(hilo_file):
    os.remove(hilo_file)
if os.path.exists(hda_file):
    os.remove(hda_file)
if os.path.exists(betfa_file):
    os.remove(betfa_file)
if os.path.exists(match_info_file):
    os.remove(match_info_file)
# if os.path.exists(error_file):
    # os.remove(error_file)

# Timeout in seconds
timeout = 60

# Define Chrome
chrome_driver = '%s\\chromedriver.exe' % curr_path
option = webdriver.ChromeOptions()
option.add_argument("--headless")
browser = webdriver.Chrome(executable_path=chrome_driver, options=option)
# browser = webdriver.Chrome(executable_path=chrome_driver)
browser.implicitly_wait(30)
browser.set_page_load_timeout(timeout)
socket.setdefaulttimeout(timeout)

# Define HKJC leagues
hkjc_league_list = '亞冠盃,俄盃,俄超,南球盃,墨西哥盃,墨西聯,墨西聯春,巴西甲,巴甲,巴聖錦標,巴西盃,德乙,德甲,德國盃,意甲,挪超,日職乙,日職聯,日超杯,日聯盃,智利甲,欧青U21外,歐國聯,歐冠盃,歐霸盃,國際友誼,比甲,法乙,法甲,澳洲甲,瑞典盃,瑞典超,美冠盃,美職業,自由盃,英冠,英甲,英聯盃,英超,英足總盃,英錦賽,荷乙,荷甲,葡盃,葡超,蘇總盃,蘇超,西甲,西乙,阿根廷盃,阿甲,韓K聯'

# Database connection
connection = None
try:
    connection = cx_Oracle.connect(
        db_user,
        db_password,
        db_dsn,
        encoding=db_encoding)
    
    # Show the version of the Oracle Database
    print('Database connected. version: %s' % connection.version)
    
    # Open db cursor
    c = connection.cursor()
    
    print('=================================================================')
    logger('I', 'Start scraping...')
    print('=================================================================')
    
    # Execute
    if mode == '1':
        scrape('id', param)

    if mode == '2':
        get_match_info(param, 'history')
        scrape('history', param)

    if mode == '3':
        get_match_info(param, 'current')
        scrape('current', param)

    if mode == '4':
        get_match_info(param, 'upcoming')
        scrape('upcoming', param)
        
    if mode == '5':
        sql = 'SELECT TO_CHAR(TO_DATE(MIN(MATCH_DATE),\'YYYY-MM-DD\')-1,\'YYYYMMDD\') FROM MATCHDAY_STATUS WHERE IS_SCRAPED=\'Y\''
        c.execute(sql)
        scrape_date = c.fetchone()[0]
        logger('I', 'Scrape date: %s' % scrape_date)
        get_match_info(scrape_date, 'history')
        scrape('history', scrape_date)

    print('=================================================================')
    logger('I','Scraping completed')
    print('=================================================================')

except cx_Oracle.Error as error:
    browser.quit()
    logger('E','Oracle error - %s' % error)
    file = codecs.open(error_file, 'a+', 'utf-8')
    file.write('Oracle error - %s\n' % (error))
    file.close()
finally:
    # release the connection
    if connection:
        browser.quit()
        # Database packages
        c.execute('BEGIN PKG_HILO.SP_HILO_MEAN_MEDIAN(0); END;')
        c.execute('BEGIN PKG_HILO.SP_HILO_MERGE2(0); END;')
        c.execute('BEGIN PKG_HDA.SP_HDA_MEAN_MEDIAN(0); END;')
        c.execute('BEGIN PKG_ASIAN.SP_ASIAN_MERGE(0); END;')
        logger('I','Database packages executed successfully')
        connection.close()

