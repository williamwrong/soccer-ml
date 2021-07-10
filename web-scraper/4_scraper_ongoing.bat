@ECHO OFF
:loop

python %cd%\scraper.py 4

timeout 300

goto loop