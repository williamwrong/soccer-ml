--------------------------------------------------------
--  DDL for Table HDA_MEAN_MEDIAN
--------------------------------------------------------

  CREATE TABLE "JW"."HDA_MEAN_MEDIAN" 
   (	"MATCH_ID" NUMBER, 
	"HANDICAP_TYPE" NUMBER, 
	"HOME_MEAN" NUMBER, 
	"HOME_MEDIAN" NUMBER, 
	"DRAW_MEAN" NUMBER, 
	"DRAW_MEDIAN" NUMBER, 
	"AWAY_MEAN" NUMBER, 
	"AWAY_MEDIAN" NUMBER, 
	"O_MACAU_H" NUMBER, 
	"O_CROWN_H" NUMBER, 
	"O_BET365_H" NUMBER, 
	"O_EASYWIN_H" NUMBER, 
	"O_WAITAK_H" NUMBER, 
	"O_MINGSING_H" NUMBER, 
	"O_10BET_H" NUMBER, 
	"O_KAMBO_H" NUMBER, 
	"O_12BET_H" NUMBER, 
	"O_LEEKEE_H" NUMBER, 
	"O_YINGYO_H" NUMBER, 
	"O_18BET_H" NUMBER, 
	"O_PINGBOK_H" NUMBER, 
	"O_HKJC_H" NUMBER, 
	"O_MACAU_D" NUMBER, 
	"O_CROWN_D" NUMBER, 
	"O_BET365_D" NUMBER, 
	"O_EASYWIN_D" NUMBER, 
	"O_WAITAK_D" NUMBER, 
	"O_MINGSING_D" NUMBER, 
	"O_10BET_D" NUMBER, 
	"O_KAMBO_D" NUMBER, 
	"O_12BET_D" NUMBER, 
	"O_LEEKEE_D" NUMBER, 
	"O_YINGYO_D" NUMBER, 
	"O_18BET_D" NUMBER, 
	"O_PINGBOK_D" NUMBER, 
	"O_HKJC_D" NUMBER, 
	"O_MACAU_A" NUMBER, 
	"O_CROWN_A" NUMBER, 
	"O_BET365_A" NUMBER, 
	"O_EASYWIN_A" NUMBER, 
	"O_WAITAK_A" NUMBER, 
	"O_MINGSING_A" NUMBER, 
	"O_10BET_A" NUMBER, 
	"O_KAMBO_A" NUMBER, 
	"O_12BET_A" NUMBER, 
	"O_LEEKEE_A" NUMBER, 
	"O_YINGYO_A" NUMBER, 
	"O_18BET_A" NUMBER, 
	"O_PINGBOK_A" NUMBER, 
	"O_HKJC_A" NUMBER
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;
