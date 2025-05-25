from pyspark.sql import functions as F
from pyspark.sql.functions import udf, first, lit, split, explode
from pyspark.sql import types as T
from pyspark.sql.window import Window

valid_types = ['df']
invalid_aspect_names = ['mergednamespacenames', 'maturityscore',
                        'isprplinkenabled', 'lhexposetotse', 'seller-selected-epid',
                        'non-domestic product', 'modified item', 'upc',
                        'producttitle', 'ean', 'savedautotagprodrefid', 'ebay product id (epid)',
                        'gtin', 'california prop 65 warning', 'catrecoscore_1',
                        'catrecoscore_2', 'catrecoscore_3', 'catrecoid_1', 'catrecoid_2',
                        'catrecoid_3', 'miscatscore', 'p2sprobability', 'miscatscore_v1',
                        'uom1', 'miscatscore_v1', 'uom2', 'mpn', 'uom3', 'productimagezoomguid',
                        'manufacturer part number', 'isbn-13', 'isbn-10', 'other part number',
                        'miscatscore_cf_v1', 'isclplinkenabled', 'oe/oem part number',
                        'features', 'model', 'ks', 'number in pack',
                        'item height', 'item length', 'item width', 'item weight',
                        'number of items in set', 'food aisle', 'width', 'length', 'items included',
                        'custom bundle', 'volume', 'period after opening (pao)', 'featured refinements',
                        'set includes',
                        ]


def lowercase_string_columns(df):
    string_cols = [item[0] for item in df.dtypes if item[1].startswith('string')]
    for col_name in string_cols:
        df = df.withColumn(col_name, F.lower(F.col(col_name)))
    return df


combineMap = udf(lambda maps: {key: f[key] for f in maps for key in f},
                 T.MapType(T.StringType(), T.StringType()))


def collect_item_aspects(session, df_items):
    df_items = df_items.select('ITEM_ID', "AUCT_END_DT").dropDuplicates().cache()

    max_auct_end_dt = df_items.agg({"AUCT_END_DT": "max"}).collect()[0][0]
    min_auct_end_dt = df_items.agg({"AUCT_END_DT": "min"}).collect()[0][0]

    # get aspects #ITEM_ASPCT_CLSSFCTN
    df_item_aspects = session.table("ACCESS_VIEWS.ITEM_ASPCT_CLSSFCTN_SAP") \
        .where((min_auct_end_dt < F.col('AUCT_END_DT')) & (F.col('AUCT_END_DT') < max_auct_end_dt)) \
        .join(df_items, on=['ITEM_ID', "AUCT_END_DT"])

    # in case of multiple values per aspect, select the one with the shortest string value
    df_item_aspects = df_item_aspects.withColumn('VALUE_LEN', F.length("ASPCT_VLU_NM").alias('VALUE_LEN'))
    row_column = F.row_number().over(Window.partitionBy("ITEM_ID", "PRDCT_ASPCT_NM")
                                     .orderBy(df_item_aspects['VALUE_LEN']))
    df_item_aspects = df_item_aspects.withColumn("row_num", row_column.alias("row_num")) \
        .where(F.col("row_num") == 1)
    df_item_aspects = lowercase_string_columns(df_item_aspects)
    df_item_aspects = df_item_aspects.where((F.col("NS_TYPE_CD").isin(valid_types))
                                            & (~F.col("PRDCT_ASPCT_NM").isin(invalid_aspect_names)))

    # group all aspect name-value rows per item into a row with a dict column
    df_item_aspects = df_item_aspects \
        .withColumn("ASPECTS", F.create_map("PRDCT_ASPCT_NM", "ASPCT_VLU_NM").alias("ASPECTS")) \
        .groupBy('ITEM_ID') \
        .agg(F.collect_list('ASPECTS').alias('ASPECTS')) \
        .select('ITEM_ID', combineMap('ASPECTS').alias('ASPECTS'))
    return df_item_aspects


def collect_item_variation_aspects(session, df_items):
    df_item_variations = df_items.select(['ITEM_ID', 'AUCT_END_DT', 'ITEM_VRTN_ID']) \
        .where(F.col('ITEM_VRTN_ID').isNotNull()) \
        .dropDuplicates().cache()

    max_auct_end_dt = df_item_variations.agg({"AUCT_END_DT": "max"}).collect()[0][0]
    min_auct_end_dt = df_item_variations.agg({"AUCT_END_DT": "min"}).collect()[0][0]

    # get all variation aspects
    df_item_vrtn_aspects = session.table("access_views.LSTG_ITEM_VRTN_TRAIT") \
        .where((min_auct_end_dt < F.col('AUCT_END_DT')) & (F.col('AUCT_END_DT') < max_auct_end_dt)) \
        .join(df_item_variations, on=['ITEM_ID', "AUCT_END_DT", 'ITEM_VRTN_ID']) \
        .dropna(subset=['TRAIT_NAME', 'TRAIT_VALUE']) \
        .withColumn('VALUE_LEN', F.length("TRAIT_VALUE").alias('VALUE_LEN'))

    # heuristic: if multiple values per aspect, keep the shortest
    row_num_col = F.row_number().over(Window.partitionBy("ITEM_ID", "ITEM_VRTN_ID", "TRAIT_NAME")
                                      .orderBy(df_item_vrtn_aspects['VALUE_LEN']))
    df_item_vrtn_aspects = df_item_vrtn_aspects.withColumn("row_num", row_num_col.alias("row_num")) \
        .where(F.col("row_num") == 1)
    df_item_vrtn_aspects = lowercase_string_columns(df_item_vrtn_aspects)

    df_item_vrtn_aspects = df_item_vrtn_aspects \
        .withColumn("VRTN_ASPECTS", F.create_map("TRAIT_NAME", "TRAIT_VALUE").alias("VRTN_ASPECTS")) \
        .groupBy("ITEM_ID", "ITEM_VRTN_ID") \
        .agg(F.collect_list('VRTN_ASPECTS').alias('VRTN_ASPECTS')) \
        .select("ITEM_ID", "ITEM_VRTN_ID", combineMap("VRTN_ASPECTS").alias("VRTN_ASPECTS"))
    return df_item_vrtn_aspects


def get_category_names(session, df_categories):
    df_categories = df_categories.select("SITE_ID", "LEAF_CATEG_ID").dropDuplicates()
    df_categ_names = session.table("ACCESS_VIEWS.DW_CATEGORY_GROUPINGS") \
        .where(F.col('LEAF_CATEG_ID') == F.col('MOVE_TO')) \
        .join(df_categories, ["SITE_ID", "LEAF_CATEG_ID"]) \
        .dropDuplicates(["SITE_ID", "LEAF_CATEG_ID"]) \
        .select("SITE_ID", "LEAF_CATEG_ID", "LEAF_CATEG_NAME", "CATEG_LVL2_ID", "CATEG_LVL2_NAME", "META_CATEG_ID", "META_CATEG_NAME")
    return df_categ_names


def get_item_data(session, config, df):
    df = df.alias("df")
    df_item_ids = df.select("ITEM_ID").dropDuplicates()
    df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
        .select("AUCT_TITL", "ITEM_ID", "AUCT_END_DT", "LEAF_CATEG_ID", "ITEM_SITE_ID", "GALLERY_URL") \
        .withColumnRenamed("ITEM_SITE_ID", "SITE_ID") \
        .withColumnRenamed("AUCT_TITL", "TITLE") \
        .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
        .join(df_item_ids, "ITEM_ID") \
        .dropDuplicates(["ITEM_ID"])
    
    df_itms = (df.join(df_item_fact, "ITEM_ID")
                  .join(get_category_names(session, df_item_fact), on=["SITE_ID", "LEAF_CATEG_ID"])  
                  .select('df.*', df_item_fact.TITLE, df_item_fact.SITE_ID, df_item_fact.LEAF_CATEG_ID, "LEAF_CATEG_NAME", "CATEG_LVL2_ID", "CATEG_LVL2_NAME", "META_CATEG_ID", "META_CATEG_NAME")
                  .join(collect_item_aspects(session, df_item_fact), on='ITEM_ID', how='left')
              )
    
    return df_itms

def get_leaf_cat_data(session, config, df):
    df = df.alias("df")
    df = df.withColumn("SITE_ID", F.lit(0))
    df_lcats = (df.join(get_category_names(session, df), on=["SITE_ID", "LEAF_CATEG_ID"])  
                  .select('df.*', "LEAF_CATEG_NAME", "CATEG_LVL2_ID", "CATEG_LVL2_NAME", "META_CATEG_ID", "META_CATEG_NAME")
              )
    return df_lcats


def limit_history_len(max1, max2, df_user_histories):
    if max1 is None:
        max_history_len = max2
    elif max2 is None:
        max_history_len = max1
    else:
        max_history_len = min(max1, max2)

    if max_history_len is not None:
        window = Window.partitionBy("BUYER_ID").orderBy(F.col("EVENT_TIMESTAMP").desc())
        df_user_histories = df_user_histories.withColumn("row_number", F.row_number().over(window)) \
            .filter(F.col("row_number") < max_history_len).drop("row_number")
    return df_user_histories


def get_vi_histories(session, config, df_users_to_consider):
    # get histories
    df_user_histories = session.table("ACCESS_VIEWS.VI_EVENT_FACT") \
        .where((F.col("SESSION_START_DT") >= config['filters']['start_dt']) &
               (F.col("SESSION_START_DT") <= config['filters']['end_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids']))) \
        .withColumnRenamed('USER_ID', 'BUYER_ID') \
        .join(df_users_to_consider, "BUYER_ID") \
        .select('GUID', 'SESSION_SKEY', 'SESSION_START_DT', 'SEQNUM', 'EVENT_TIMESTAMP', 'BUYER_ID',
                'ITEM_ID', 'ITEM_VRTN_ID', 'CURPRICE', 'QTYS', 'QTYA', 'SRC_LINK_ID', 'SRC_MODULE_ID', 'SRC_PAGE_ID')
    df_user_histories = limit_history_len(config['event_limits']['max_vi'],
                                          config['event_limits']['max_events'],
                                          df_user_histories)

    # get item_fact details
    df_item_ids = df_user_histories.select("ITEM_ID").dropDuplicates()
    df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
        .select("AUCT_TITL", "ITEM_ID", "AUCT_END_DT", "LEAF_CATEG_ID", "ITEM_SITE_ID", "GALLERY_URL") \
        .withColumnRenamed("ITEM_SITE_ID", "SITE_ID") \
        .withColumnRenamed("AUCT_TITL", "TITLE") \
        .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
        .join(df_item_ids, "ITEM_ID") \
        .dropDuplicates(["ITEM_ID"])
    df_item_fact = lowercase_string_columns(df_item_fact)

    # join everything together
    df_final = df_user_histories \
        .join(df_item_fact.select('ITEM_ID', 'TITLE', 'SITE_ID', "LEAF_CATEG_ID"), on='ITEM_ID') \
        .join(get_category_names(session, df_item_fact), on=["SITE_ID", "LEAF_CATEG_ID"]) \
        .join(collect_item_aspects(session, df_item_fact), on='ITEM_ID', how='left') \
        .withColumn('EVENT_TYPE', lit('VI'))

    # seems that some rows are corrupt in the data and contain nulls in must-have columns
    df_final_vi = df_final.filter(F.col('ITEM_ID').isNotNull() & F.col('SITE_ID').isNotNull() &
                                  F.col('LEAF_CATEG_ID').isNotNull() & F.col('TITLE').isNotNull() &
                                  F.col('LEAF_CATEG_NAME').isNotNull())
    df_final_vi = lowercase_string_columns(df_final_vi)

    return df_final_vi

def get_minimal_vi_histories(session, config, df_users_to_consider):
    # get histories
    df_user_histories = session.table("ACCESS_VIEWS.VI_EVENT_FACT") \
        .where((F.col("SESSION_START_DT") >= config['filters']['start_dt']) &
               (F.col("SESSION_START_DT") <= config['filters']['end_dt']) &
               (F.col("SITE_ID").isin(config['filters']['site_ids']))) \
        .withColumnRenamed('USER_ID', 'BUYER_ID') \
        .join(df_users_to_consider, "BUYER_ID") \
        .select('EVENT_TIMESTAMP', 'BUYER_ID', 'ITEM_ID')

    df_user_histories = limit_history_len(config['event_limits']['max_vi'],
                                          config['event_limits']['max_events'],
                                          df_user_histories)

    # get item_fact details
    df_item_ids = df_user_histories.select("ITEM_ID").dropDuplicates()
    df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
        .select("AUCT_TITL", "ITEM_ID", "AUCT_END_DT", "LEAF_CATEG_ID", "ITEM_SITE_ID") \
        .withColumnRenamed("ITEM_SITE_ID", "SITE_ID") \
        .withColumnRenamed("AUCT_TITL", "TITLE") \
        .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
        .join(df_item_ids, "ITEM_ID") \
        .dropDuplicates(["ITEM_ID"])

    df_item_fact = lowercase_string_columns(df_item_fact)

    # join everything together
    df_final = df_user_histories \
        .join(df_item_fact.select('ITEM_ID', 'TITLE', 'SITE_ID', "LEAF_CATEG_ID"), on='ITEM_ID') \
        .join(get_category_names(session, df_item_fact), on=["SITE_ID", "LEAF_CATEG_ID"]) \
        .withColumn('EVENT_TYPE', lit('VI'))

    # seems that some rows are corrupt in the data and contain nulls in must-have columns
    df_final_vi = df_final.filter(F.col('ITEM_ID').isNotNull() & 
                                  F.col('SITE_ID').isNotNull() &
                                  F.col('LEAF_CATEG_ID').isNotNull() & 
                                  F.col('TITLE').isNotNull() &
                                  F.col('LEAF_CATEG_NAME').isNotNull())

    return df_final_vi


def get_purchase_histories(session, config, df_users_to_consider):
    
    df_user_histories = session.table("access_views.DW_CHECKOUT_TRANS") \
        .where((F.col("CREATED_DT") >= config['filters']['start_dt'])
               & (F.col("CREATED_DT") <= config['filters']['end_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids']))) \
        .join(df_users_to_consider, "BUYER_ID") \
        .withColumnRenamed("CREATED_TIME", "EVENT_TIMESTAMP") \
        .select('BUYER_ID', 'ITEM_ID', 'ITEM_VRTN_ID', 'AUCT_END_DT', 'TRANSACTION_ID',
                'LEAF_CATEG_ID', 'SITE_ID', 'SELLER_ID', 'ITEM_PRICE', "EVENT_TIMESTAMP", "CREATED_DT", "QUANTITY")
    df_user_histories = limit_history_len(config['event_limits']['max_purchases'],
                                          config['event_limits']['max_events'],
                                          df_user_histories)
    

    # 4. Get more information: item returns, condition, title category names
    df_returned_items = session.table("access_views.DW_SHPMT_RTRN_LINE_ITEM") \
        .withColumnRenamed('TRANS_ID', 'TRANSACTION_ID') \
        .withColumn("item_returned", F.lit(1))

    df_item_ids = df_user_histories.select("ITEM_ID").dropDuplicates()
    df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
        .select("AUCT_TITL", "ITEM_ID", "AUCT_END_DT", "ITEM_SITE_ID", "GALLERY_URL") \
        .withColumnRenamed("ITEM_SITE_ID", "SITE_ID") \
        .withColumnRenamed("AUCT_TITL", "TITLE") \
        .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
        .join(df_item_ids, "ITEM_ID") \
        .dropDuplicates(["ITEM_ID", "SITE_ID", 'AUCT_END_DT'])
    df_item_fact = lowercase_string_columns(df_item_fact)

    # 5. join everything together
    df_final = df_user_histories \
        .join(collect_item_aspects(session, df_user_histories), on=['ITEM_ID'], how='left') \
        .join(collect_item_variation_aspects(session, df_user_histories), on=['ITEM_ID', 'ITEM_VRTN_ID'], how='left') \
        .join(df_returned_items.select('ITEM_ID', 'TRANSACTION_ID', 'ITEM_RETURNED'),
              on=['ITEM_ID', 'TRANSACTION_ID'], how='left') \
        .join(df_item_fact, on=['ITEM_ID', 'SITE_ID', 'AUCT_END_DT']) \
        .join(get_category_names(session, df_user_histories), on=['LEAF_CATEG_ID', 'SITE_ID']) \
        .fillna(0, subset=['ITEM_RETURNED'])

    
    df_final = df_final.where(df_final.ITEM_RETURNED == 0)

    df_final_purchases = df_final.select("ITEM_ID", "SITE_ID", "LEAF_CATEG_ID",
                                         "ITEM_VRTN_ID", "VRTN_ASPECTS",
                                         F.col('ITEM_PRICE').alias("PRICE"),
                                         "QUANTITY", 
                                         "LEAF_CATEG_NAME", "ASPECTS", "BUYER_ID", "EVENT_TIMESTAMP",
                                         "CATEG_LVL2_ID", "META_CATEG_ID", "GALLERY_URL", "TITLE", "CATEG_LVL2_NAME", "META_CATEG_NAME") \
        .withColumn('EVENT_TYPE', lit('PURCHASE')) \
        .withColumn('GMV', F.col("PRICE") * F.col("QUANTITY"))
    return lowercase_string_columns(df_final_purchases)


def get_minimal_purchase_histories(session, config, df_users_to_consider):
    
    df_user_histories = session.table("access_views.DW_CHECKOUT_TRANS") \
        .where((F.col("CREATED_DT") >= config['filters']['start_dt'])
               & (F.col("CREATED_DT") <= config['filters']['end_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids']))) \
        .join(df_users_to_consider, "BUYER_ID") \
        .withColumnRenamed("CREATED_TIME", "EVENT_TIMESTAMP") \
        .select('BUYER_ID', 'ITEM_ID', 'AUCT_END_DT', 'TRANSACTION_ID',
                'LEAF_CATEG_ID', 'SITE_ID', 'ITEM_PRICE', "EVENT_TIMESTAMP", "CREATED_DT", "QUANTITY")
    df_user_histories = limit_history_len(config['event_limits']['max_purchases'],
                                          config['event_limits']['max_events'],
                                          df_user_histories)
    

    # 4. Get more information: item returns, condition, title category names
    df_returned_items = session.table("access_views.DW_SHPMT_RTRN_LINE_ITEM") \
        .withColumnRenamed('TRANS_ID', 'TRANSACTION_ID') \
        .withColumn("item_returned", F.lit(1))

    df_item_ids = df_user_histories.select("ITEM_ID").dropDuplicates()
    df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
        .select("AUCT_TITL", "ITEM_ID", "AUCT_END_DT", "ITEM_SITE_ID") \
        .withColumnRenamed("ITEM_SITE_ID", "SITE_ID") \
        .withColumnRenamed("AUCT_TITL", "TITLE") \
        .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
        .join(df_item_ids, "ITEM_ID") \
        .dropDuplicates(["ITEM_ID", "SITE_ID", 'AUCT_END_DT'])
    df_item_fact = lowercase_string_columns(df_item_fact)

    # 5. join everything together
    df_final = df_user_histories \
        .join(df_returned_items.select('ITEM_ID', 'TRANSACTION_ID', 'ITEM_RETURNED'),
              on=['ITEM_ID', 'TRANSACTION_ID'], how='left') \
        .join(df_item_fact, on=['ITEM_ID', 'SITE_ID', 'AUCT_END_DT']) \
        .join(get_category_names(session, df_user_histories), on=['LEAF_CATEG_ID', 'SITE_ID']) \
        .fillna(0, subset=['ITEM_RETURNED'])

    
    df_final = df_final.where(df_final.ITEM_RETURNED == 0)

    df_final_purchases = df_final.select("ITEM_ID", "SITE_ID", "LEAF_CATEG_ID",
                                         F.col('ITEM_PRICE').alias("PRICE"),
                                         "QUANTITY", 
                                         "LEAF_CATEG_NAME", "BUYER_ID", "EVENT_TIMESTAMP",
                                         "CATEG_LVL2_ID", "META_CATEG_ID", "TITLE", "CATEG_LVL2_NAME", "META_CATEG_NAME") \
        .withColumn('EVENT_TYPE', lit('PURCHASE')) \
        .withColumn('GMV', F.col("PRICE") * F.col("QUANTITY"))
    return lowercase_string_columns(df_final_purchases)


def get_hp_events(session, config, df_users_to_consider):
    df_hp_events = session.table("access_views.HP_EVENT_FACT") \
        .where((F.col("DT") >= config['filters']['start_dt'].replace('-', '')) &
               (F.col("DT") <= config['filters']['end_dt'].replace('-', ''))) \
        .withColumnRenamed('signedin_uid', 'BUYER_ID') \
        .join(df_users_to_consider, "BUYER_ID") \
        .groupBy("session_start_dt", "pageci", "BUYER_ID") \
        .agg(first("EVENT_TIMESTAMP").alias("EVENT_TIMESTAMP")) \
        .select('BUYER_ID', 'EVENT_TIMESTAMP', 'SESSION_START_DT') \
        .withColumn("EVENT_TYPE", F.lit("HP"))
    return limit_history_len(config['event_limits']['max_hp_events'],
                             config['event_limits']['max_events'], df_hp_events)


def get_search_histories(session, config, df_users_to_consider):
    df_user_histories = session.table("access_views.SRCH_SRP_EVENT_FACT") \
        .where((F.col("SESSION_START_DT") >= config['filters']['start_dt']) &
               (F.col("SESSION_START_DT") <= config['filters']['end_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids'])) &
               ((F.col('CLEAN_ASPECTS').isNotNull() & (F.col('CLEAN_ASPECTS') != '')) |
                (F.col('CLEAN_QUERY').isNotNull() & (F.col('CLEAN_QUERY') != '')))) \
        .withColumnRenamed('USER_ID', 'BUYER_ID') \
        .join(df_users_to_consider, "BUYER_ID") \
        .select("SITE_ID", "BUYER_ID",
                F.col("CATEGORY_ID").alias("LEAF_CATEG_ID"),
                "SESSION_START_DT", "GUID", "SESSION_SKEY", "COBRAND", "SEQNUM", "EVENT_TIMESTAMP",
                F.col("UPPER_PRICE_LIMITED").alias("PRICE"),
                F.col("CLEAN_QUERY").alias("TITLE"),
                F.col("CLEAN_ASPECTS").alias("SEARCH_QUERY_ASPECTS"),
                "PAGINATION_NUM", "VI_CNT", "VI_DTLS", "ITEM_LIST") \
        .withColumn("EVENT_TYPE", F.lit("SEARCH"))
    df_user_histories = limit_history_len(config['event_limits']['max_searches'],
                                          config['event_limits']['max_events'],
                                          df_user_histories)

    # get SRP item result titles
    df_srp_items = df_user_histories.select("SESSION_START_DT", "SITE_ID", "GUID", "SESSION_SKEY", "COBRAND",
                                            "SEQNUM", F.col("ITEM_LIST").alias("ITEM_ID"))
    df_srp_items = df_srp_items.withColumn("ITEM_ID", explode(split("ITEM_ID", ',')))

    if config['collect_srp_item_titles']:
        df_item_ids = df_srp_items.select("ITEM_ID").dropDuplicates()
        df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
            .select(F.col("AUCT_TITL").alias("TITLE"), "ITEM_ID", "GALLERY_URL") \
            .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
            .join(df_item_ids, "ITEM_ID") \
            .dropDuplicates(["ITEM_ID"])
        df_srp_items = df_srp_items.join(df_item_fact, on='ITEM_ID')
    else:
        print('Skipping srp item titles!')

    group_key = ["SESSION_START_DT", "SITE_ID", "GUID", "SESSION_SKEY", "COBRAND", "SEQNUM"]
    grouped_cols = [c for c in df_srp_items.columns if c not in group_key]
    df_srp_items = df_srp_items.select(F.struct(*grouped_cols).alias('ITEM'), *group_key)
    df_srp_items = df_srp_items.groupBy(group_key).agg(F.collect_list("ITEM").alias("ITEM_LIST"))

    df_final_search = df_user_histories \
        .drop('ITEM_LIST') \
        .join(get_category_names(session, df_user_histories), on=['LEAF_CATEG_ID', 'SITE_ID']) \
        .join(df_srp_items, on=["SESSION_START_DT", "SITE_ID", "GUID", "SESSION_SKEY", "COBRAND", "SEQNUM"]) \
        .drop('row_number')
    return lowercase_string_columns(df_final_search)

def get_minimal_search_histories(session, config, df_users_to_consider):
    # Get search histories
    df_user_histories = session.table("access_views.SRCH_SRP_EVENT_FACT") \
        .where((F.col("SESSION_START_DT") >= config['filters']['start_dt']) &
               (F.col("SESSION_START_DT") <= config['filters']['end_dt']) &
               (F.col("SITE_ID").isin(config['filters']['site_ids'])) &
               ((F.col('CLEAN_ASPECTS').isNotNull() & (F.col('CLEAN_ASPECTS') != '')) |
                (F.col('CLEAN_QUERY').isNotNull() & (F.col('CLEAN_QUERY') != '')))) \
        .withColumnRenamed('USER_ID', 'BUYER_ID') \
        .join(df_users_to_consider, "BUYER_ID") \
        .select('EVENT_TIMESTAMP', 'BUYER_ID', 
                F.col("CATEGORY_ID").alias("LEAF_CATEG_ID"),
                F.col("CLEAN_QUERY").alias("TITLE"),
                F.col("SITE_ID"))  # Include SITE_ID here

    # Apply limits to search history length
    df_user_histories = limit_history_len(config['event_limits']['max_searches'],
                                          config['event_limits']['max_events'],
                                          df_user_histories)

    # Join everything together
    df_final = df_user_histories \
        .join(get_category_names(session, df_user_histories), on=["SITE_ID", "LEAF_CATEG_ID"]) \
        .withColumn('EVENT_TYPE', lit('SEARCH')) \
        .select('EVENT_TIMESTAMP', 'BUYER_ID', 'LEAF_CATEG_ID', 'TITLE', 'SITE_ID', 'EVENT_TYPE',
                "LEAF_CATEG_NAME", "CATEG_LVL2_ID", "META_CATEG_ID", "CATEG_LVL2_NAME", "META_CATEG_NAME")  # Select necessary columns

    # Filter out rows with nulls
    df_final_search = df_final.filter(F.col('TITLE').isNotNull() & 
                                      F.col('SITE_ID').isNotNull() & 
                                      F.col('LEAF_CATEG_ID').isNotNull())
    return lowercase_string_columns(df_final_search) 


def get_minimal_watch_histories(session, config, df_users_to_consider):
    # Get watch histories from bbowac_event_fact with event_type_desc='Watch'
    df_user_histories = session.table("access_views.bbowac_event_fact") \
        .where((F.col("session_start_dt") >= config['filters']['start_dt']) &  
               (F.col("session_start_dt") <= config['filters']['end_dt']) &
               (F.upper(F.col("site_id")).isin(config['filters']['site_ids'])) &  
               (F.col("event_type_desc") == 'Watch') &  # Filter where event_type_desc is 'Watch'
               (F.col("bot_session") == 0))  # Exclude bot sessions

    # Rename columns
    df_user_histories = df_user_histories \
        .withColumnRenamed('buyer_id', 'BUYER_ID') \
        .withColumnRenamed('site_id', 'SITE_ID') \
        .withColumnRenamed('item_id', 'ITEM_ID') \
        .withColumnRenamed('variation_id', 'VARIATION_ID') \
        .withColumnRenamed('leaf_categ_id', 'LEAF_CATEG_ID')  \
        .withColumnRenamed('event_timestamp', 'EVENT_TIMESTAMP')  \
        .select('EVENT_TIMESTAMP', 'BUYER_ID', 'ITEM_ID', 'VARIATION_ID', 'LEAF_CATEG_ID', 'SITE_ID') 
    
    df_user_histories = df_user_histories.join(df_users_to_consider, "BUYER_ID") \
        .select('EVENT_TIMESTAMP', 'BUYER_ID', 'ITEM_ID')

    # Limit history length using max_watches
    df_user_histories = limit_history_len(config['event_limits']['max_watches'],  
                                          config['event_limits']['max_events'],
                                          df_user_histories) 
    
    df_item_ids = df_user_histories.select("ITEM_ID").dropDuplicates() 
    
    df_item_fact = session.table("access_views.DW_LSTG_ITEM") \
        .select("AUCT_TITL", "ITEM_ID", "AUCT_END_DT", "LEAF_CATEG_ID", "ITEM_SITE_ID") \
        .withColumnRenamed("ITEM_SITE_ID", "SITE_ID") \
        .withColumnRenamed("AUCT_TITL", "TITLE") \
        .where(F.col('AUCT_END_DT') >= config['filters']['start_dt']) \
        .join(df_item_ids, "ITEM_ID") \
        .dropDuplicates(["ITEM_ID"])
    
    df_item_fact = lowercase_string_columns(df_item_fact) 
   
    df_final = df_user_histories \
        .join(df_item_fact.select('ITEM_ID', 'TITLE', 'SITE_ID', "LEAF_CATEG_ID"), on='ITEM_ID') \
        .join(get_category_names(session, df_item_fact), on=["SITE_ID", "LEAF_CATEG_ID"]) \
        .withColumn('EVENT_TYPE', lit('WATCH')) 
   
    df_final_watch = df_final.filter(F.col('ITEM_ID').isNotNull() & 
                                  F.col('SITE_ID').isNotNull() &
                                  F.col('LEAF_CATEG_ID').isNotNull() & 
                                  F.col('TITLE').isNotNull() &
                                  F.col('LEAF_CATEG_NAME').isNotNull())

    return lowercase_string_columns(df_final_watch)

#not relevant
def get_minimal_watch_histories_watch_metric_item(session, config, df_users_to_consider):
    # Get watch histories
    df_user_histories = session.table("ACCESS_VIEWS.WATCH_METRIC_ITEM") \
        .where((F.col("SRC_CRE_DT") >= config['filters']['start_dt']) &
               (F.col("SRC_CRE_DT") <= config['filters']['end_dt']) &
               (F.col("SITE_ID").isin(config['filters']['site_ids'])) &
               (F.col("DEL_GUID").isNull()))  # Removed AUCT_END_DT condition

    # Rename USER_ID to BUYER_ID and join with users to consider
    df_user_histories = df_user_histories \
        .withColumnRenamed('USER_ID', 'BUYER_ID') \
        .join(df_users_to_consider, "BUYER_ID") \
        .select('SRC_CRE_DATE', 'BUYER_ID', 'ITEM_ID', 'VARIATION_ID', 'LEAF_CATEG_ID', 'SITE_ID') \
        .withColumnRenamed('SRC_CRE_DATE', 'EVENT_TIMESTAMP')  # Renaming SRC_CRE_DT to EVENT_TIMESTAMP
    
    # Limit history length using max_watches
    df_user_histories = limit_history_len(config['event_limits']['max_watches'],  # Changed here
                                          config['event_limits']['max_events'],
                                          df_user_histories)

    # Join category names and select necessary columns
    df_final = df_user_histories \
        .join(get_category_names(session, df_user_histories), on=["SITE_ID", "LEAF_CATEG_ID"]) \
        .withColumn('EVENT_TYPE', lit('WATCH')) \
        .select('EVENT_TIMESTAMP', 'BUYER_ID', 'LEAF_CATEG_ID', 'ITEM_ID', 'VARIATION_ID', 
                'SITE_ID', 'EVENT_TYPE', 
                "LEAF_CATEG_NAME", "CATEG_LVL2_ID", "META_CATEG_ID", 
                "CATEG_LVL2_NAME", "META_CATEG_NAME")  # Select necessary columns

    # Filter out corrupt data
    df_final_watch = df_final.filter(F.col('ITEM_ID').isNotNull() & 
                                     F.col('SITE_ID').isNotNull() &
                                     F.col('LEAF_CATEG_ID').isNotNull() & 
                                     F.col('LEAF_CATEG_NAME').isNotNull())
    
    # Return final DataFrame with lowercase string columns
    return lowercase_string_columns(df_final_watch)



def user_histories_pyspark(session, config):
    df_users_to_consider = session.table("access_views.DW_CHECKOUT_TRANS") \
        .where((F.col("CREATED_DT") >= config['filters']['start_dt'])
               & (F.col("CREATED_DT") <= config['filters']['end_dt'])
               & (F.col("SITE_ID").isin(config['filters']['site_ids']))) \
        .groupBy(F.col("BUYER_ID")) \
        .agg(F.count("CREATED_DT").alias("transaction_date_count")) \
        .where(F.col("transaction_date_count") >= config['filters']['min_purchase_dates']) \
        .select("BUYER_ID") \
        .sample(fraction=config['filters']['sample_fraction'], seed=42)

    # collect all required data
    all_data = []
    if config['event_limits']['max_purchases'] != 0:
        all_data.append(get_purchase_histories(session, config, df_users_to_consider))
        if config['filters']['min_purchase_dates'] > 0:
            # if we use purchases, we only want other events for users with a purchase
            df_users_to_consider = all_data[-1].select('BUYER_ID').dropDuplicates()
    if config['event_limits']['max_vi'] != 0:
        all_data.append(get_vi_histories(session, config, df_users_to_consider))

    if config['event_limits']['max_searches'] != 0:
        all_data.append(get_search_histories(session, config, df_users_to_consider))

    if config['event_limits']['max_hp_events'] != 0:
        all_data.append(get_hp_events(session, config, df_users_to_consider))

    # Union dataframes, take care of missing fields
    all_fields = set(field.name for df in all_data for field in df.schema.fields)
    df_final = None
    for j, df in enumerate(all_data):
        df_fields = set(field.name for field in df.schema.fields)
        for field in all_fields - df_fields:
            df = df.withColumn(field, F.lit(None))
        df_final = df if df_final is None else df_final.unionByName(df)

    df_final = limit_history_len(config['event_limits']['max_events'], None, df_final)

    # group events by user id
    group_key = ['BUYER_ID']
    grouped_cols = [c for c in df_final.columns if c not in group_key]
    df_final = df_final.select(F.struct(*grouped_cols).alias('event'), *group_key)
    df_final = df_final.groupBy(group_key).agg(F.collect_list("event").alias("events"))

    return df_final