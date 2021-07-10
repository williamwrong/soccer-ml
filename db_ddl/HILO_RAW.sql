--------------------------------------------------------
--  DDL for Table HILO_RAW
--------------------------------------------------------

  CREATE TABLE "JW"."HILO_RAW" 
   (	"MATCH_ID" NUMBER, 
	"BOOKMAKER" VARCHAR2(20 BYTE), 
	"LINE" VARCHAR2(20 BYTE), 
	"START_HI" NUMBER, 
	"START_HANDICAP" NUMBER, 
	"START_LO" NUMBER, 
	"END_HI" NUMBER, 
	"END_HANDICAP" NUMBER, 
	"END_LO" NUMBER, 
	"IS_MERGED" VARCHAR2(1 BYTE)
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;
