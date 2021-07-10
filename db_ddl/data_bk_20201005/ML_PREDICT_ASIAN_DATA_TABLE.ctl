OPTIONS (ERRORS=151)
LOAD DATA 
INFILE 'D:\github-workspace\fbp-jupyter-notebook\backup_20201005\ML_PREDICT_ASIAN_DATA_TABLE.ldr' "str '{EOL}'"
APPEND
CONTINUEIF NEXT(1:1) = '#'
INTO TABLE "JW"."ML_PREDICT_ASIAN"
FIELDS TERMINATED BY'|'
OPTIONALLY ENCLOSED BY '"' AND '"'
TRAILING NULLCOLS ( 
"ML_TYPE" CHAR (20),
"MATCH_ID" ,
"MATCH_DATETIME" DATE "YYYY-MM-DD HH24:MI:SS" ,
"LEAGUE" CHAR (50),
"HOME_TEAM" CHAR (50),
"AWAY_TEAM" CHAR (50),
"STR_A_MACAU_HDC" ,
"STR_A_MACAU_H" ,
"STR_A_MACAU_A" ,
"STR_HDA_MACAU_H" ,
"STR_HDA_MACAU_D" ,
"STR_HDA_MACAU_A" ,
"STR_A_HKJC_HDC" ,
"STR_A_HKJC_H" ,
"STR_A_HKJC_A" ,
"STR_HDA_HKJC_H" ,
"STR_HDA_HKJC_D" ,
"STR_HDA_HKJC_A" ,
"FAV_PROB" ,
"UND_PROB" ,
"MATCH_DATE" )
