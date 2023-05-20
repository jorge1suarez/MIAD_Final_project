DROP TABLE "ts_gspc";
CREATE TABLE "ts_gspc" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_aapl";
CREATE TABLE "ts_aapl" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_msft";
CREATE TABLE "ts_msft" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_goog";
CREATE TABLE "ts_goog" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_googl";
CREATE TABLE "ts_googl" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_tsla";
CREATE TABLE "ts_tsla" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_amzn";
CREATE TABLE "ts_amzn" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_brka";
CREATE TABLE "ts_brka" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_brkb";
CREATE TABLE "ts_brkb" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_nvda";
CREATE TABLE "ts_nvda" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_meta";
CREATE TABLE "ts_meta" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_unh";
CREATE TABLE "ts_unh" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_bzf";
CREATE TABLE "ts_bzf" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_ngf";
CREATE TABLE "ts_ngf" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_gcf";
CREATE TABLE "ts_gcf" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_eurusdx";
CREATE TABLE "ts_eurusdx" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_vix";
CREATE TABLE "ts_vix" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_ixic";
CREATE TABLE "ts_ixic" ( 
	"date" date primary key, 
	"open" real, 
	"high" real, 
	"low" real, 
	"close" real, 
	"adj_close" real, 
	"volume" real 
); 

DROP TABLE "ts_effr";
CREATE TABLE "ts_effr" ( 
	"date" date primary key, 
	"value" real 
); 

DROP TABLE "ts_csushpisa";
CREATE TABLE "ts_csushpisa" ( 
	"date" date primary key, 
	"value" real 
); 

DROP TABLE "ts_gdp";
CREATE TABLE "ts_gdp" ( 
	"date" date primary key, 
	"value" real 
); 

DROP TABLE "ts_cpiaucsl";
CREATE TABLE "ts_cpiaucsl" ( 
	"date" date primary key, 
	"value" real 
); 

DROP TABLE "ts_cpilfesl";
CREATE TABLE "ts_cpilfesl" ( 
	"date" date primary key, 
	"value" real 
); 

DROP TABLE "ts_gspc_pred";
CREATE TABLE "ts_gspc_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_aapl_pred";
CREATE TABLE "ts_aapl_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_msft_pred";
CREATE TABLE "ts_msft_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_goog_pred";
CREATE TABLE "ts_goog_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_googl_pred";
CREATE TABLE "ts_googl_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_tsla_pred";
CREATE TABLE "ts_tsla_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_amzn_pred";
CREATE TABLE "ts_amzn_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_brka_pred";
CREATE TABLE "ts_brka_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_brkb_pred";
CREATE TABLE "ts_brkb_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_nvda_pred";
CREATE TABLE "ts_nvda_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_meta_pred";
CREATE TABLE "ts_meta_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_unh_pred";
CREATE TABLE "ts_unh_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_bzf_pred";
CREATE TABLE "ts_bzf_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_ngf_pred";
CREATE TABLE "ts_ngf_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_gcf_pred";
CREATE TABLE "ts_gcf_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_eurusdx_pred";
CREATE TABLE "ts_eurusdx_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_vix_pred";
CREATE TABLE "ts_vix_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_ixic_pred";
CREATE TABLE "ts_ixic_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_effr_pred";
CREATE TABLE "ts_effr_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_csushpisa_pred";
CREATE TABLE "ts_csushpisa_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_gdp_pred";
CREATE TABLE "ts_gdp_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_cpiaucsl_pred";
CREATE TABLE "ts_cpiaucsl_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

DROP TABLE "ts_cpilfesl_pred";
CREATE TABLE "ts_cpilfesl_pred" ( 
	"date" date primary key, 
	"mean" real, 
	"min" real, 
	"max" real, 
	"q2_5" real, 
	"q5" real, 
	"q25" real, 
	"q50" real, 
	"q75" real, 
	"q95" real, 
	"q97_5" real 
); 

