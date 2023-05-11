
-- DROP TABLE "ts_gspc";
-- DROP TABLE "ts_aapl";
-- DROP TABLE "ts_effr";
-- DROP TABLE "ts_msft";
-- DROP TABLE "ts_goog";
-- DROP TABLE "ts_googl";
-- DROP TABLE "ts_tsla";
-- DROP TABLE "ts_amzn";
-- DROP TABLE "ts_brka";
-- DROP TABLE "ts_brkb";
-- DROP TABLE "ts_nvda";
-- DROP TABLE "ts_meta";
-- DROP TABLE "ts_unh";
-- DROP TABLE "ts_bzf";
-- DROP TABLE "ts_ngf";
-- DROP TABLE "ts_gcf";
-- DROP TABLE "ts_eurusdx";
-- DROP TABLE "ts_vix";
-- DROP TABLE "ts_ixic";
-- DROP TABLE "ts_csushpisa";
-- DROP TABLE "ts_gdp";
-- DROP TABLE "ts_cpiaucsl";
-- DROP TABLE "ts_cpilfesl";


CREATE TABLE "ts_gspc" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_aapl" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_msft" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_goog" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_googl" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_tsla" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_amzn" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_brka" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);


CREATE TABLE "ts_brkb" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);


CREATE TABLE "ts_nvda" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_meta" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_unh" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);


CREATE TABLE "ts_bzf" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_ngf" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_gcf" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_eurusdx" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_vix" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);

CREATE TABLE "ts_ixic" (
  "date" date primary key,
  "open" real,
  "high" real,
  "low" real,
  "close" real,
  "adj_close" real,
  "volume" real
);


CREATE TABLE "ts_effr" (
  "date" date primary key,
  "value" real
);

CREATE TABLE "ts_csushpisa" (
  "date" date primary key,
  "value" real
);

CREATE TABLE "ts_gdp" (
  "date" date primary key,
  "value" real
);


CREATE TABLE "ts_cpiaucsl" (
  "date" date primary key,
  "value" real
);

CREATE TABLE "ts_cpilfesl" (
  "date" date primary key,
  "value" real
);
