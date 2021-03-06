--------------------------------------------------------
--  DDL for Table HDA_RAW
--------------------------------------------------------

  CREATE TABLE "JW"."HDA_RAW" 
   (	"MATCH_ID" NUMBER, 
	"BOOKMAKER" VARCHAR2(50 BYTE), 
	"HANDICAP_TYPE" NUMBER, 
	"HOME_ODD" NUMBER, 
	"DRAW_ODD" NUMBER, 
	"AWAY_ODD" NUMBER, 
	"HOME_WIN_RATE" NUMBER, 
	"DRAW_WIN_RATE" NUMBER, 
	"AWAY_WIN_RATE" NUMBER, 
	"RETURN_RATE" NUMBER, 
	"HOME_KELLY" NUMBER, 
	"DRAW_KELLY" NUMBER, 
	"AWAY_KELLY" NUMBER, 
	"IS_MERGED" VARCHAR2(1 BYTE)
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;
