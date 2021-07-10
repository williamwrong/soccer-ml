@ECHO OFF

echo ==================== Asian025 ====================
python %cd%\AsianML_025.py

echo ==================== Asian075 ====================
python %cd%\AsianML_075.py

echo ==================== Twist ====================
python %cd%\AsianML_Twist.py

echo ==================== Twist2 ====================
python %cd%\AsianML_Twist2.py

echo ==================== JCTwist ====================
python %cd%\AsianML_JCTwist.py

echo ==================== HDA ====================
python %cd%\AsianML_HDA.py

echo ==================== gsheets ====================
python %cd%\gsheets_asian.py
