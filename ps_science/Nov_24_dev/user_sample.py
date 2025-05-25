from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def user_hist_purchases(session, config):
    df_prch = (session.table("ACCESS_VIEWS.DW_CHECKOUT_TRANS") 
                .where((F.col("CREATED_DT") >= config['filters']['start_dt'])
               & (F.col("CREATED_DT") < config['sampling']['ref_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids'])))
                .withColumnRenamed("CREATED_TIME", "EVENT_TIMESTAMP"))
    return df_prch

def sample_purchases(session, config):
    df_prch = user_hist_purchases(session, config) 
    df_prch = (df_prch.select('BUYER_ID', 'LEAF_CATEG_ID').dropDuplicates()
                      .withColumn("EVENT_TIMESTAMP", F.lit(config['sampling']['ref_dt'])))
    df_prch = df_prch.sample(False, config['sampling']['sample_ratio'])
    if 'max_samples' in config['sampling'] and config['sampling']['max_samples'] > 0:
        df_prch = df_prch.limit(config['sampling']['max_samples'])
    return df_prch


label_func = F.udf(lambda x: 1 if x > 0 else 0, IntegerType())

def label_repurchases(session, config, df_up):
    df_prch = (session.table("ACCESS_VIEWS.DW_CHECKOUT_TRANS") 
        .where((F.col("CREATED_DT") > config['sampling']['ref_dt'])
               & (F.col("CREATED_DT") <= config['filters']['end_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids'])))
        .withColumnRenamed("CREATED_TIME", "EVENT_TIMESTAMP")
        .select('BUYER_ID', 'LEAF_CATEG_ID', "EVENT_TIMESTAMP"))
    df_up = (df_up.join(df_prch, (df_up.BUYER_ID == df_prch.BUYER_ID) & 
                        (df_up.LEAF_CATEG_ID == df_prch.LEAF_CATEG_ID) &
                        (df_up.EVENT_TIMESTAMP < df_prch.EVENT_TIMESTAMP), how="left")
                        .groupBy([df_up.BUYER_ID, df_up.LEAF_CATEG_ID, df_up.EVENT_TIMESTAMP])
                        .agg(F.count(df_prch.EVENT_TIMESTAMP).alias("num_repurchases"))
                        .fillna(0)
                        .withColumn("label", label_func(F.col("num_repurchases"))))
    return df_up                     
                      
def sample_data(session, config):
    df_prch = sample_purchases(session, config)
    df_sample = label_repurchases(session, config, df_prch)
    return df_sample


def calc_leaf_cat_priors(session, config):
    df_prch = user_hist_purchases(session, config)
    df_lcat_prch = (df_prch.select("BUYER_ID", "LEAF_CATEG_ID", "EVENT_TIMESTAMP")
                      .groupBy(["BUYER_ID", "LEAF_CATEG_ID"]).agg(F.count("EVENT_TIMESTAMP").alias("num_purchases"))
                      .where(F.col("num_purchases") > 1) #repurchase
                      .groupBy("LEAF_CATEG_ID").agg(F.mean(F.col("num_purchases")).alias("exp_num_repurchases")))
    return df_lcat_prch
