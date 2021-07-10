@ECHO OFF
:loop

python %cd%\scraper.py 5

timeout 60

goto loop