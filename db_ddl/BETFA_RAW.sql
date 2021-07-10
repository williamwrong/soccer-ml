--------------------------------------------------------
--  DDL for Table BETFA_RAW
--------------------------------------------------------

  CREATE TABLE "JW"."BETFA_RAW" 
   (	"MATCH_ID" NUMBER, 
	"ALL_HOUSE_HOME_IDX" NUMBER, 
	"ALL_HOUSE_DRAW_IDX" NUMBER, 
	"ALL_HOUSE_AWAY_IDX" NUMBER, 
	"ALL_HOUSE_HOME_WIN_RATE" NUMBER, 
	"ALL_HOUSE_DRAW_WIN_RATE" NUMBER, 
	"ALL_HOUSE_AWAY_WIN_RATE" NUMBER, 
	"ALL_HOUSE_RETURN_RATE" NUMBER, 
	"BETFA_HOME_IDX" NUMBER, 
	"BETFA_DRAW_IDX" NUMBER, 
	"BETFA_AWAY_IDX" NUMBER, 
	"BETFA_HOME_WIN_RATE" NUMBER, 
	"BETFA_DRAW_WIN_RATE" NUMBER, 
	"BETFA_AWAY_WIN_RATE" NUMBER, 
	"BETFA_RETURN_RATE" NUMBER, 
	"BETFA_HOME_VOLUME" NUMBER, 
	"BETFA_DRAW_VOLUME" NUMBER, 
	"BETFA_AWAY_VOLUME" NUMBER, 
	"BETFA_HOME_TRADE_RATE" NUMBER, 
	"BETFA_DRAW_TRADE_RATE" NUMBER, 
	"BETFA_AWAY_TRADE_RATE" NUMBER, 
	"BETFA_HOME_HANDICAP" NUMBER, 
	"BETFA_HANDICAP" VARCHAR2(20 BYTE), 
	"BETFA_AWAY_HANDICAP" NUMBER, 
	"BETFA_HOME_PL" NUMBER, 
	"BETFA_DRAW_PL" NUMBER, 
	"BETFA_AWAY_PL" NUMBER, 
	"BETFA_HOME_PL_IDX" NUMBER, 
	"BETFA_DRAW_PL_IDX" NUMBER, 
	"BETFA_AWAY_PL_IDX" NUMBER, 
	"BETFA_HOME_HC_IDX" NUMBER, 
	"BETFA_DRAW_HC_IDX" NUMBER, 
	"BETFA_AWAY_HC_IDX" NUMBER, 
	"BETFA_HOME_BUYER" NUMBER, 
	"BETFA_DRAW_BUYER" NUMBER, 
	"BETFA_AWAY_BUYER" NUMBER, 
	"BETFA_HOME_SELLER_IDX" NUMBER, 
	"BETFA_DRAW_SELLER_IDX" NUMBER, 
	"BETFA_AWAY_SELLER_IDX" NUMBER, 
	"TOTAL_HOME_BUYER" NUMBER, 
	"TOTAL_DRAW_BUYER" NUMBER, 
	"TOTAL_AWAY_BUYER" NUMBER, 
	"TOTAL_HOME_SELLER" NUMBER, 
	"TOTAL_DRAW_SELLER" NUMBER, 
	"TOTAL_AWAY_SELLER" NUMBER
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;
