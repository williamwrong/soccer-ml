@ECHO OFF
REM :loop

REM SET /P date="Scrape match date: "

REM del "%cd%\data\%date%_*" /s /f /q
REM start cmd /k "python %cd%\scraper.py 2 %date%"
python %cd%\scraper.py 2

pause

REM goto loop
