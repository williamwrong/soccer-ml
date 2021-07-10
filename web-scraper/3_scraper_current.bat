@ECHO OFF
:loop

python %cd%\scraper.py 3

timeout 600

goto loop