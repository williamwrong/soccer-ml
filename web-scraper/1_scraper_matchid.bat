@ECHO OFF
:loop

SET /P matchid="Scrape match id: "

REM del "%cd%\data\%date%_*" /s /f /q
start cmd /k "python %cd%\scraper.py 1 %matchid%"

REM pause

goto loop
