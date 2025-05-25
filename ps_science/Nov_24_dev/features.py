from pyspark.sql import functions as F
import feature_helpers as fe
from pyspark.sql.types import FloatType
import numpy as np
from pyspark.sql import Window  

def filter_inactive_buyers(session, config, df_ft, df_vi):
    df_have_activity = (df_ft.join(df_vi, df_ft.BUYER_ID == df_vi.BUYER_ID)
                             .where(df_vi.EVENT_TIMESTAMP > config['sampling']['ref_dt'])
                             .select(df_ft.BUYER_ID).dropDuplicates())
    df_fltr = df_ft.join(df_have_activity, "BUYER_ID")
    return df_fltr
                        
def add_feature(df_base, df_feature, feature_name=None, join_key="ITEM_ID", na_value=None, how="inner"):
    df_res = df_base.join(df_feature, on=["BUYER_ID", join_key, "EVENT_TIMESTAMP"], how=how)
   # if na_value:
    if na_value is not None and feature_name:
        df_res = df_res.fillna(na_value, subset=[feature_name])
    return df_res

#---------------------------------------------------
def _leaf_condition(df1, df2, cat_type="LEAF"):
    if cat_type == "LEAF":
        return (df1.LEAF_CATEG_ID == df2.LEAF_CATEG_ID)
    elif cat_type == "LVL2":
        return (df1.CATEG_LVL2_ID == df2.CATEG_LVL2_ID)
    else:
        return (df1.META_CATEG_ID == df2.META_CATEG_ID) 

def _leaf_id_col(df, cat_type="LEAF"):
    if cat_type == "LEAF":
        return df.LEAF_CATEG_ID
    elif cat_type == "LVL2":
        return df.CATEG_LVL2_ID
    else:
        return df.META_CATEG_ID
    
def _leaf_name_col(df, cat_type="LEAF"):
    if cat_type == "LEAF":
        return df.LEAF_CATEG_NAME
    elif cat_type == "LVL2":
        return df.CATEG_LVL2_NAME
    else:
        return df.META_CATEG_NAME 
#----------------------------------------------------------------------------------------------   
def f_weighted_num_actions_from_cat_in_last_x_days(df_base, df_prch, cat_type="LEAF", action_type="purchase",
                                                   time_windows=[2, 7, 30, 90, 180, 360], weights=[0.35, 0.25, 0.2, 0.1, 0.06, 0.04]): 
    
    # Define the final feature name for weighted actions
    feature_name = f"f_weighted_num_{action_type}s_from_{cat_type}_cat"
    
    # Initialize a column to hold the cumulative weighted sum
    df_base = df_base.withColumn(feature_name, F.lit(0))
    
    # Calculate each time-window feature and apply the corresponding weight
    for i, days in enumerate(time_windows):
        df_window = f_num_actions_from_cat_in_last_x_days(df_base, df_prch, cat_type=cat_type, days=days, action_type=action_type)
       
        
        # Generate the specific feature name from the base function for this time window
        window_feature_name = f"f_num_{action_type}s_from_{cat_type}_cat_in_last_{days}_days"
        

        df_window = (df_window
                     .withColumnRenamed("BUYER_ID", "window_BUYER_ID")
                     .withColumnRenamed("EVENT_TIMESTAMP", "window_EVENT_TIMESTAMP"))
        
        # Join the result to the base DataFrame and apply the weight
        df_base = (df_base
                   .join(
                       df_window.select("window_BUYER_ID", _leaf_id_col(df_window, cat_type), "window_EVENT_TIMESTAMP", window_feature_name),
                       on=[
                           df_base["BUYER_ID"] == df_window["window_BUYER_ID"],
                           _leaf_condition(df_base, df_window, cat_type),
                           df_base["EVENT_TIMESTAMP"] == df_window["window_EVENT_TIMESTAMP"]
                       ], 
                       how="left"
                   )
                   .select(
                       df_base["BUYER_ID"],
                       _leaf_id_col(df_base, cat_type),
                       df_base["EVENT_TIMESTAMP"],
                       df_base[feature_name],
                       F.coalesce(F.col(window_feature_name), F.lit(0)).alias(f"weighted_{window_feature_name}")
                   ))
        

        weighted_column = F.coalesce(F.col(f"weighted_{window_feature_name}"), F.lit(0)) * weights[i]
        
        # Accumulate the weighted sum into the main feature column
        df_base = df_base.withColumn(feature_name, F.col(feature_name) + weighted_column)
      #  df_base.show()
    # Return only necessary columns: identifier and the final weighted feature
    df_res = df_base.select("BUYER_ID", _leaf_id_col(df_base, cat_type), "EVENT_TIMESTAMP", feature_name) 
    
    df_res_no_dups = df_res.distinct()
    return df_res_no_dups
#----------------------------------------------------------------------------------------------   
def f_actions_ratio_from_cat_in_last_x_days(df_base, df_vi, df_prch, cat_type="LEAF", days=30, action_types=["view", "purchase"]):
    # Generate dynamic feature names based on action types
    action_1_feature = f"f_num_{action_types[0]}s_from_{cat_type}_cat_in_last_{days}_days"
    action_2_feature = f"f_num_{action_types[1]}s_from_{cat_type}_cat_in_last_{days}_days"
    ratio_feature = f"f_{action_types[0]}_to_{action_types[1]}_ratio_from_{cat_type}_cat_in_last_{days}_days"
    
    # Calculate counts for the first action type 
    df_vi_counts = f_num_actions_from_cat_in_last_x_days(df_base, df_vi, cat_type=cat_type, days=days, action_type=action_types[0])
    
    # Calculate counts for the second action type 
    df_prch_counts = f_num_actions_from_cat_in_last_x_days(df_base, df_prch, cat_type=cat_type, days=days, action_type=action_types[1])
    
    # Join action counts on buyer ID, category ID, and event timestamp
    df_ratio = (df_vi_counts
                .join(df_prch_counts, 
                      on=[df_vi_counts.BUYER_ID == df_prch_counts.BUYER_ID,
                          _leaf_condition(df_vi_counts, df_prch_counts, cat_type),
                          df_vi_counts.EVENT_TIMESTAMP == df_prch_counts.EVENT_TIMESTAMP],
                      how="left")
                .select(df_vi_counts.BUYER_ID, 
                        _leaf_id_col(df_vi_counts, cat_type),
                        df_vi_counts.EVENT_TIMESTAMP,
                        F.coalesce(df_vi_counts[action_1_feature], F.lit(0)).alias(action_1_feature),
                        F.coalesce(df_prch_counts[action_2_feature], F.lit(0)).alias(action_2_feature))
               )
    
    # Calculate the ratio
    df_ratio = df_ratio.withColumn(ratio_feature,
                                   F.when(F.col(action_2_feature) == 0, F.lit(-1))  # Avoid division by zero
                                   .otherwise(F.col(action_1_feature) / F.col(action_2_feature)))
    # Dropping the specified columns
    df_ratio = df_ratio.drop(action_1_feature , action_2_feature)

    
    return df_ratio
#----------------------------------------------------------------------------------------------

def f_norm_num_actions_from_cat_in_last_x_days(df_base, df_action, cat_type="LEAF", days=1, action_type="purchase"):
    # Step 1: Calculate the number of actions from the specific category type
    cat_actions_df = f_num_actions_from_cat_in_last_x_days(df_base, df_action, cat_type=cat_type, days=days, action_type=action_type)
    
    feature_name = f"f_norm_num_{action_type}s_from_{cat_type}_cat_in_last_{days}_days"
    
    # Step 2: Calculate total actions across all categories of the same type (not limited to a specific leaf)
    total_actions = f"f_total_{action_type}s_in_last_{days}_days" 
    df_total_actions = (df_base.join(df_action, 
                                      (df_base.BUYER_ID == df_action.BUYER_ID) & 
                                      (F.datediff(df_base.EVENT_TIMESTAMP, df_action.EVENT_TIMESTAMP) > 0) &
                                      (F.datediff(df_base.EVENT_TIMESTAMP, df_action.EVENT_TIMESTAMP) <= days), 
                                      how="left")
                        .groupBy(df_base.BUYER_ID, df_base.EVENT_TIMESTAMP)
                        .agg(F.count(df_action.EVENT_TIMESTAMP).alias(total_actions))
                        .fillna(0, subset=[total_actions])
    )

    # Step 3: Join the category-specific actions with the total actions
    df_result = cat_actions_df.join(df_total_actions, 
                                     ["BUYER_ID", "EVENT_TIMESTAMP"], 
                                     "left").fillna(0)

    # Step 4: Normalize the category-specific actions by the total actions
    df_result = df_result.withColumn(feature_name,
                                      F.when(df_result[total_actions] > 0,  
                                              F.round(                                                 df_result[f"f_num_{action_type}s_from_{cat_type}_cat_in_last_{days}_days"] / df_result[total_actions], 
                                                  5)  
                                              ).otherwise(-1))

    df_result = df_result.drop(f"f_num_{action_type}s_from_{cat_type}_cat_in_last_{days}_days", total_actions)

    return df_result



#----------------------------------------------------------------------------------------------        
    
def f_user_vs_mean_cat_actions_in_last_x_days(df_base, df_action, cat_type="LEAF", days=30, action_type="purchase"):
    feature_name = f"f_user_vs_mean_{action_type}_from_{cat_type}_cat_in_last_{days}_days"
    
    # Step 1: Calculate user activity in the last 'days' period
    df_user_actions = (
        df_base.join(df_action, 
            (df_base.BUYER_ID == df_action.BUYER_ID) & 
            _leaf_condition(df_base, df_action, cat_type) &
            (F.datediff(df_base.EVENT_TIMESTAMP, df_action.EVENT_TIMESTAMP) > 0) &
            (F.datediff(df_base.EVENT_TIMESTAMP, df_action.EVENT_TIMESTAMP) <= days), 
            how="left")
        .groupBy(df_base.BUYER_ID, _leaf_id_col(df_base, cat_type), df_base.EVENT_TIMESTAMP)
        .agg(F.count(df_action.EVENT_TIMESTAMP).alias(f"user_{action_type}_count"))
    )
    
    # Step 2: Calculate population mean and stddev only for values > 0
    df_population_stats = (
        df_user_actions.groupBy(_leaf_id_col(df_user_actions, cat_type))
        .agg(
            F.mean(f"user_{action_type}_count").alias("population_mean"),
            F.stddev(f"user_{action_type}_count").alias("population_stddev")
        )
    )
    
    # Step 3: Join user actions with population stats and calculate z-score
    df_result = (
        df_user_actions.join(
            df_population_stats, 
            on=_leaf_condition(df_user_actions, df_population_stats, cat_type), 
            how="left"
        )
        .withColumn(
            feature_name, 
            F.when(
                (F.col("population_stddev") > 0) & (F.col("population_stddev").isNotNull()), 
                (F.col(f"user_{action_type}_count") - F.col("population_mean")) / F.col("population_stddev")
            ).otherwise(0)
        )
        .fillna(0,subset=[feature_name] )
    ).select(
            df_user_actions.BUYER_ID,
            _leaf_id_col(df_user_actions, cat_type),
            df_user_actions.EVENT_TIMESTAMP,
            F.col(feature_name))
    
    df_res = df_result.distinct() 

    return df_res
    

#----------------------------------------------------------------------------------
# Function to calculate category diversity entropy
def add_category_entropy(df_base, df_action, action_type="purchase", cat_type="LEAF"):
    df_action_base = (
        df_base
        .join(df_action, 
              (df_base.BUYER_ID == df_action.BUYER_ID) & 
              _leaf_condition(df_base, df_action, cat_type), 
              how="left")
        .filter(F.datediff(df_base.EVENT_TIMESTAMP, df_action.EVENT_TIMESTAMP) > 0)
        .select(
            df_action.BUYER_ID,
            df_action.ITEM_ID, 
            df_action.EVENT_TIMESTAMP, 
            _leaf_id_col(df_action, cat_type))) 

    # Step 3: Count number of actions (purchases/views) per user per category
    df_counts = (
        df_action_base
        .groupBy(df_action_base.BUYER_ID, _leaf_id_col(df_action, cat_type))
        .agg(F.count("*").alias("cat_action_count"))
    )

    # Step 4: Calculate total number of actions per user
    df_total = (
        df_action_base
        .groupBy(df_action_base.BUYER_ID)
        .agg(F.count("*").alias("total_action_count"))
    )

    # Step 5: Join category counts with total counts
    df_joined = (
        df_counts
        .join(df_total, on="BUYER_ID", how="inner")
        .withColumn("proportion",
            F.when(F.col("total_action_count") > 0, F.col("cat_action_count") / F.col("total_action_count")).otherwise(-1)
        )
    )
    
    # Step 6: Group by user and calculate entropy for each user
    df_entropy = (
        df_joined
        .groupBy("BUYER_ID")
        .agg(F.collect_list("proportion").alias("proportions"))
    )

    # Step 7: Apply entropy function to the list of proportions
    df_entropy = (
        df_entropy
        .withColumn("category_entropy", fe.entropy(F.col("proportions")))
        .select("BUYER_ID", "category_entropy")
    )

    return df_entropy 

def f_category_entropy(df_base, df_action, action_type="purchase", cat_type="LEAF"):
    # Calculate category entropy for the given category type and action type
    df_entropy = add_category_entropy(df_base, df_action, action_type=action_type, cat_type=cat_type)
    
    df_res = (
        df_base
        .join(df_entropy, on="BUYER_ID", how="left")
        .withColumnRenamed("category_entropy", f"f_entropy_for_{cat_type}_cat_{action_type}s")
        .select(
            df_base.BUYER_ID,
            _leaf_id_col(df_base, cat_type),
            df_base.EVENT_TIMESTAMP,
            F.col(f"f_entropy_for_{cat_type}_cat_{action_type}s"))
    )
    df_res = df_res.fillna(-1, subset=[f"f_entropy_for_{cat_type}_cat_{action_type}s"])
    
    df_res_no_dups = df_res.distinct() 
    return df_res_no_dups

#----------------------------------------------------------------------------------

# Helper method to compute views and time-related features from last purchase
def _compute_view_based_features(df_prch, df_vi, df_base, cat_type="LEAF"):
    # Step 1: Join purchases with views and calculate the time difference
    df_view_join = (df_prch
                    .join(df_vi, 
                          (df_prch.BUYER_ID == df_vi.BUYER_ID) & 
                          _leaf_condition(df_prch, df_vi, cat_type), how="left")
                    .select(
                        df_vi.BUYER_ID,
                        df_vi.ITEM_ID,
                        df_vi.EVENT_TIMESTAMP.alias("view_timestamp"), 
                        df_prch.EVENT_TIMESTAMP.alias("purchase_timestamp"), 
                        _leaf_id_col(df_prch, cat_type) 
                    )
                    .withColumn("time_diff", F.datediff(F.col("view_timestamp"), F.col("purchase_timestamp")))
                    .filter(F.col("time_diff") > 0)  
                   )

    # Step 4: Join df_view_join with df_base to filter views that happen after the base event timestamp
    df_view_join = (df_view_join
                    .join(df_base, 
                          (df_view_join.BUYER_ID == df_base.BUYER_ID) & 
                          _leaf_condition(df_view_join, df_base, cat_type), how="inner")
                     .filter((F.datediff(df_base.EVENT_TIMESTAMP, F.col("view_timestamp")) > 0))
                    .select(
                        df_view_join.BUYER_ID,
                        df_view_join.ITEM_ID,
                        df_view_join.view_timestamp,
                        df_view_join.purchase_timestamp,
                        _leaf_id_col(df_view_join, cat_type),
                        df_view_join.time_diff
                           
                    )
                   )

    # New Step: Ensure each view is associated with the closest preceding purchase
    # Step 5: Use a window function to rank views per buyer and category
    view_ranked = df_view_join.withColumn(
        "rank", 
        F.row_number().over(Window.partitionBy(
            df_view_join.BUYER_ID,
            _leaf_id_col(df_view_join, cat_type),
            df_view_join.view_timestamp
        ).orderBy(F.desc("purchase_timestamp")))
    )

    view_single_assigned = view_ranked.filter(F.col("rank") == 1).drop("rank")

    # Step 7: Assign row number to each view-purchase pair, ordering by purchase date descending per buyer have last purchase.
    window_spec = Window.partitionBy(view_single_assigned.BUYER_ID, _leaf_id_col(df_prch, cat_type) ).orderBy(F.desc("purchase_timestamp"))
    view_single_assigned = view_single_assigned.withColumn("rn", F.rank().over(window_spec))
    
    view_single = view_single_assigned.filter(F.col("rn") == 1).drop("rn")
    
    # Step 5: Group by user, category, and purchase timestamp to calculate features
    df_grouped = (view_single.groupBy(
                        view_single.BUYER_ID,  
                        _leaf_id_col(view_single, cat_type), 
                        view_single.purchase_timestamp)
                   .agg(
                        F.count("view_timestamp").alias("num_views"),  # Count views since last purchase
                        F.max("view_timestamp").alias("last_view_timestamp"),  # Last view timestamp
                        F.max("time_diff").alias("days_since_last_view")  # Time since the last view to purchase
                    )
                 )

    # Step 6: Filter rows where the last view is not null
    df_grouped = df_grouped.filter(F.col("last_view_timestamp").isNotNull())
    return df_grouped

# Feature generation function to add the number of views and days since last view
def f_views_since_last_purchase_in_cat(df_base, df_prch, df_vi, cat_type="LEAF"):
    # Step 1: Compute the view-based features
    df_view_features = _compute_view_based_features(df_prch, df_vi, df_base, cat_type)

    # Step 2: Join the computed features with the base dataset
    df_res = (df_base.join(df_view_features, 
                           (df_base.BUYER_ID == df_view_features.BUYER_ID) & 
                           (F.datediff(df_base.EVENT_TIMESTAMP, F.col("last_view_timestamp")) > 0)&
                           _leaf_condition(df_base, df_view_features, cat_type), 
                           how="left")
              .select(
                  df_base.BUYER_ID,
                  _leaf_id_col(df_base, cat_type),
                  df_base.EVENT_TIMESTAMP,
                  df_view_features.num_views.alias(f"f_num_views_since_last_purchase_in_{cat_type}_cat"),
                  df_view_features.days_since_last_view.alias(f"f_time_from_last_view_to_last_purchase_in_{cat_type}_cat")))
     
    df_res = df_res.fillna(0, subset=[f"f_num_views_since_last_purchase_in_{cat_type}_cat"])
    df_res = df_res.fillna(-1, subset=[f"f_time_from_last_view_to_last_purchase_in_{cat_type}_cat"])
    
    df_res_no_dups = df_res.distinct()

    return df_res_no_dups 
#----------------------------------------------------------------------------------- 

def _num_views_from_purchase_in_cat(df_prch, df_vi, df_base, cat_type="LEAF"):
    # Step 1: Get unique purchase timestamps per buyer and category
    purchase_df = df_prch.select("BUYER_ID", _leaf_id_col(df_prch, cat_type), "EVENT_TIMESTAMP").distinct()
    
    # Step 2: Create a window to identify the next purchase timestamp for each buyer and category
    window_spec = Window.partitionBy("BUYER_ID", _leaf_id_col(df_prch, cat_type)).orderBy(F.col("EVENT_TIMESTAMP").asc())
    
    # Add the next purchase timestamp using lead
    purchase_df = purchase_df.withColumn("next_purchase_timestamp", F.lead("EVENT_TIMESTAMP", 1).over(window_spec))

    # Step 3: Join views with purchases to find views that fall between purchases
    view_purchase_joined = df_vi.join(
        purchase_df,
        (df_vi.BUYER_ID == purchase_df.BUYER_ID) & _leaf_condition(purchase_df, df_vi, cat_type), how="left"
    ).select(
        df_vi.BUYER_ID,
        df_vi.ITEM_ID,
        df_vi.EVENT_TIMESTAMP.alias("view_timestamp"),  # Rename view timestamp for clarity
        purchase_df.EVENT_TIMESTAMP.alias("purchase_timestamp"),  # Rename purchase timestamp for clarity
        _leaf_id_col(purchase_df, cat_type),  
        purchase_df.next_purchase_timestamp
    )

    # Step 4: Filter views to ensure each view is associated with the closest past purchase
    view_purchase_filtered = view_purchase_joined.filter(
        (view_purchase_joined.view_timestamp > view_purchase_joined.purchase_timestamp) & 
        (view_purchase_joined.view_timestamp <= F.coalesce(view_purchase_joined.next_purchase_timestamp, F.lit('9999-12-31')))
    )
    
    # Step 5: Use a window function to rank views per buyer and category and purchase -- a specific view associated to 1 purchase.
    view_ranked = view_purchase_filtered.withColumn(
        "rank",
        F.row_number().over(Window.partitionBy(view_purchase_filtered.BUYER_ID, _leaf_id_col(view_purchase_filtered, cat_type),
                                               view_purchase_filtered.view_timestamp )
                             .orderBy(F.col("purchase_timestamp").desc()))  # Rank by the closest purchase
    )
    
    # Step 6: Keep only the highest-ranked view (the closest to the purchase)
    view_single_assigned = view_ranked.filter(F.col("rank") == 1)
    
    # Step 7: Select the desired columns and add time_diff
    df_res = view_single_assigned.select(
        "BUYER_ID",
        "ITEM_ID",
        _leaf_id_col(view_single_assigned, cat_type),
        F.col("purchase_timestamp"),
        F.col("view_timestamp"),
        F.datediff(F.col("view_timestamp"), F.col("purchase_timestamp")).alias("time_diff")
    )

    # Step 8: Join with df_base and select the relevant columns
    df_res = (df_res.join(df_base, 
                          (df_res["BUYER_ID"] == df_base["BUYER_ID"]) &  
                          _leaf_condition(df_res, df_base, cat_type), 
                          how="inner")  # Use inner join to filter based on conditions
                  .filter((F.datediff(df_base.EVENT_TIMESTAMP, df_res.view_timestamp) > 0) & 
                    (F.datediff(df_base.EVENT_TIMESTAMP, df_res.purchase_timestamp) > 0)) 
                  .select(
                      df_res.BUYER_ID,
                      _leaf_id_col(df_res, cat_type),
                      df_base.EVENT_TIMESTAMP,
                      df_res.purchase_timestamp,
                      df_res.view_timestamp,  # Select view timestamp without alias
                      df_res.time_diff.alias("days_diff")  # Alias for time difference
                  )
              )
    
    # Step 9: Count the views between purchases and calculate max days
    df_view_count = df_res.groupBy("BUYER_ID", _leaf_id_col(df_res, cat_type),"EVENT_TIMESTAMP", "purchase_timestamp") \
                          .agg(F.count(F.col("view_timestamp")).alias("view_count"),
                               F.max(F.col("days_diff")).alias("max_days_diff"))

    # Step 10: Add new feature (views / max_days_diff)
    df_view_count = df_view_count.withColumn(
    "view_count_per_max_days",
    F.when(F.col("max_days_diff") > 0, F.col("view_count") / F.col("max_days_diff")).otherwise(F.lit(0))
)

    return df_view_count

def f_num_views_from_purchase_in_cat(df_base, df_prch, df_vi, cat_type="LEAF"):
    # Step 1: Get the number of views between purchases for the given category type
    df_view_counts = _num_views_from_purchase_in_cat(df_prch, df_vi, df_base, cat_type)

    # Step 2: Join df_base with df_view_counts to add the new features
    df_res = (df_base.join(df_view_counts, 
                           (df_base.BUYER_ID == df_view_counts.BUYER_ID) & 
                           (df_base.EVENT_TIMESTAMP == df_view_counts.EVENT_TIMESTAMP)& 
                           _leaf_condition(df_base, df_view_counts, cat_type), 
                           how="left").groupBy(df_base.BUYER_ID, _leaf_id_col(df_base, cat_type),df_base.EVENT_TIMESTAMP) \
                                       .agg(F.collect_list(F.col("view_count")).alias("view_count_array"),
                                            F.collect_list(F.col("view_count_per_max_days")).alias("view_count_per_max_days_array"))
                     .select(df_base.BUYER_ID, 
                             _leaf_id_col(df_base, cat_type),
                             df_base.EVENT_TIMESTAMP,
                             F.col("view_count_array"), 
                             F.col("view_count_per_max_days_array")) 
             )
    
    # Step 3: Calculate features based on view counts (min, max, mean)
    df_res_features = (df_res.withColumn(f"f_min_num_views_between_purchases_in_{cat_type}_cat", 
                            F.array_min(F.col("view_count_array")))  # Min value of the array
                     .withColumn(f"f_max_num_views_between_purchases_in_{cat_type}_cat", 
                            F.array_max(F.col("view_count_array")))  # Max value of the array
                     .withColumn(f"f_mean_num_views_between_purchases_in_{cat_type}_cat", 
                            fe.array_mean(F.col("view_count_array")))  # Mean value of the array
                     .withColumn(f"f_min_views_between_purchases_per_days_in_{cat_type}_cat", 
                            F.array_min(F.col("view_count_per_max_days_array")))  # Min of views/max_days ratio
                     .withColumn(f"f_max_views_between_purchases_per_days_in_{cat_type}_cat", 
                            F.array_max(F.col("view_count_per_max_days_array")))  # Max of views/max_days ratio
                     .withColumn(f"f_mean_views_between_purchases_per_days_in_{cat_type}_cat", 
                            fe.array_mean(F.col("view_count_per_max_days_array")))  # Mean of views/max_days ratio
                     .fillna(0, subset=[f"f_min_num_views_between_purchases_in_{cat_type}_cat", 
                                         f"f_max_num_views_between_purchases_in_{cat_type}_cat", 
                                         f"f_mean_num_views_between_purchases_in_{cat_type}_cat",
                                         f"f_min_views_between_purchases_per_days_in_{cat_type}_cat", 
                                         f"f_max_views_between_purchases_per_days_in_{cat_type}_cat",
                                         f"f_mean_views_between_purchases_per_days_in_{cat_type}_cat"])  
                     
                     .drop("view_count_array", "view_count_per_max_days_array")
          )

    return df_res_features

#------------------------------------------------------------------------------------
def _first_view_after_purchase(df_prch, df_vi, df_base, cat_type="LEAF"):
    # Combined Step 1 and Step 2: Join purchases and views, filter, group by, and aggregate
    df_res = (df_prch.join(df_vi, 
                           (df_prch.BUYER_ID == df_vi.BUYER_ID) & _leaf_condition(df_prch, df_vi, cat_type), how="left")
                      .withColumn("time_diff", F.datediff(df_vi.EVENT_TIMESTAMP, df_prch.EVENT_TIMESTAMP)).filter(F.col("time_diff") > 0)
                      .groupBy(df_prch.BUYER_ID, _leaf_id_col(df_prch, cat_type), df_prch.EVENT_TIMESTAMP.alias("purchase_timestamp"))
                      .agg(
                          F.min(df_vi.EVENT_TIMESTAMP).alias("first_view_timestamp"),
                          F.min("time_diff").alias("days_diff")
                      )
                      .filter(F.col("first_view_timestamp").isNotNull()))

    # Step 3: Use window function to ensure only the closest view is kept - same view can be assigned to multiple purchases so we take the closest purchase by days diff 
    window_spec = Window.partitionBy(df_res.BUYER_ID, _leaf_id_col(df_res, cat_type), df_res.first_view_timestamp).orderBy(F.col("days_diff").asc())  
    # Rank based on the closest purchase to each view timestamp
    df_res = (df_res.withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1)  # Keep only the closest view (rank == 1)
              .drop("rank"))

    # Step 4: Join with df_base and ensure base.EVENT_TIMESTAMP > first_view_timestamp
    df_res = (df_res.join(df_base, (df_res["BUYER_ID"] == df_base["BUYER_ID"]) &  _leaf_condition(df_res, df_base, cat_type), how="inner")
              .filter(F.datediff(df_base.EVENT_TIMESTAMP, df_res.first_view_timestamp) > 0)
              .select(df_res.BUYER_ID, _leaf_id_col(df_res, cat_type), df_res.purchase_timestamp, df_res.first_view_timestamp, df_res.days_diff))

    return df_res  

def f_time_since_first_view_after_purchase_in_cat(df_base, df_prch, df_vi, cat_type="LEAF"):
    # Step 1: Get the first view after each purchase
    df_first_view = _first_view_after_purchase(df_prch, df_vi, df_base, cat_type)
    
    # Step 2: Join with df_base and perform aggregation directly, calculating min, max, and mean days_diff in a single step
    df_res = df_base.join(df_first_view, 
                           (df_base.BUYER_ID == df_first_view.BUYER_ID) & 
                           _leaf_condition(df_base, df_first_view, cat_type), how="left").select(df_base.BUYER_ID, 
                             _leaf_id_col(df_base, cat_type), 
                             df_base.EVENT_TIMESTAMP,
                             F.col("days_diff"))  
  
    
    df_res_agg= df_res.groupBy(df_res.BUYER_ID, 
                       _leaf_id_col(df_res, cat_type), 
                       df_base.EVENT_TIMESTAMP).agg(
                  F.array_min(F.collect_list("days_diff")).alias(f"f_min_time_since_first_view_after_purchase_in_{cat_type}_cat"),
                  F.array_max(F.collect_list("days_diff")).alias(f"f_max_time_since_first_view_after_purchase_in_{cat_type}_cat"),
                  fe.array_mean(F.collect_list("days_diff")).alias(f"f_mean_time_since_first_view_after_purchase_in_{cat_type}_cat")
              ).fillna(-1, subset=[
                  f"f_min_time_since_first_view_after_purchase_in_{cat_type}_cat", 
                  f"f_max_time_since_first_view_after_purchase_in_{cat_type}_cat", 
                  f"f_mean_time_since_first_view_after_purchase_in_{cat_type}_cat"
              ])
          
    
    
    return df_res_agg


#---------------------------------------------------------------------------------------------------------
def f_time_since_last_action_from_cat_in_days(df_base, df_action, cat_type="LEAF", action_type="purchase"):
    feature_name = f"f_time_since_last_{action_type}_from_{cat_type}_cat"
    df_res = (df_base.join(df_action, (df_base.BUYER_ID == df_action.BUYER_ID) & _leaf_condition(df_base, df_action, cat_type)  & 
                          (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) > 0), how="left")
                     .groupBy(df_base.BUYER_ID, _leaf_id_col(df_base, cat_type), df_base.EVENT_TIMESTAMP)
                     .agg(F.datediff(df_base.EVENT_TIMESTAMP, F.max(df_action.EVENT_TIMESTAMP)).alias(feature_name))
                     .fillna(-1, subset=[feature_name])
     )
    return df_res


def f_num_actions_from_cat_in_last_x_days(df_base, df_action, cat_type="LEAF", days=1, action_type="purchase"):
    feature_name = f"f_num_{action_type}s_from_{cat_type}_cat_in_last_{days}_days"
    df_res = (df_base.join(df_action, (df_base.BUYER_ID == df_action.BUYER_ID) & _leaf_condition(df_base, df_action, cat_type) &
                       (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) > 0) &
                       (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) <= days), how="left")
                  .groupBy(df_base.BUYER_ID, _leaf_id_col(df_base, cat_type), df_base.EVENT_TIMESTAMP)
                  .agg(F.count(df_action.EVENT_TIMESTAMP).alias(feature_name))
                  .fillna(0, subset=[feature_name])
     )
    return df_res
    
    
def f_gmv_from_cat_in_last_x_days(df_base, df_prch, cat_type="LEAF", days=1):
    feature_names = {
        "sum": f"f_sum_gmv_from_{cat_type}_cat_in_last_{days}_days",
        "mean": f"f_mean_gmv_from_{cat_type}_cat_in_last_{days}_days",
        "max": f"f_max_gmv_from_{cat_type}_cat_in_last_{days}_days",
        "min": f"f_min_gmv_from_{cat_type}_cat_in_last_{days}_days",
        "var": f"f_var_gmv_from_{cat_type}_cat_in_last_{days}_days"
    }
    df_res = (df_base.join(df_prch, (df_base.BUYER_ID == df_prch.BUYER_ID) & _leaf_condition(df_base, df_prch, cat_type) &
                       (F.datediff(df_base.EVENT_TIMESTAMP,df_prch.EVENT_TIMESTAMP) > 0) &
                       (F.datediff(df_base.EVENT_TIMESTAMP,df_prch.EVENT_TIMESTAMP) <= days), how="left")
                  .groupBy(df_base.BUYER_ID, _leaf_id_col(df_base, cat_type), df_base.EVENT_TIMESTAMP)
                  .agg(F.sum(df_prch.GMV).alias(feature_names["sum"]), F.mean(df_prch.GMV).alias(feature_names["mean"]),
                       F.max(df_prch.GMV).alias(feature_names["max"]), F.min(df_prch.GMV).alias(feature_names["min"]),
                       F.variance(df_prch.GMV).alias(feature_names["var"]))
                  .fillna(0, subset=feature_names["sum"])
                  .fillna(0, subset=feature_names["mean"])
                  .fillna(-1, subset=feature_names["max"])
                  .fillna(-1, subset=feature_names["min"])
                  .fillna(-1, subset=feature_names["var"])
     )
    return df_res
    
    
def _cat_sim_to_hist_actions(df_base, df_action, cat_type="LEAF", days=1):
    leaf_id_col = _leaf_id_col(df_base, cat_type)
    leaf_name_col = _leaf_name_col(df_base, cat_type)
    df_res = (df_base.join(df_action, (df_base.BUYER_ID == df_action.BUYER_ID) & (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) > 0) &
                       (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) <= days))
                 .groupBy(df_base.BUYER_ID, leaf_id_col, leaf_name_col, df_base.EVENT_TIMESTAMP)
                 .agg(F.collect_list(_leaf_name_col(df_action, cat_type)).alias("hist_cats"))
                 .withColumn("hist_cats_cosine", fe.cosine_sim_to_target(leaf_name_col, F.col("hist_cats")))
                 .withColumn("hist_cats_jaccard", fe.jaccard_sim_to_target(leaf_name_col, F.col("hist_cats")))
                 .drop("hist_cats").drop(leaf_name_col)
                 
     )
    return df_res


def _title_sim_to_hist_actions(df_base, df_action, days=1):
    df_res = (df_base.join(df_action, (df_base.BUYER_ID == df_action.BUYER_ID) & (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) > 0) &
                       (F.datediff(df_base.EVENT_TIMESTAMP,df_action.EVENT_TIMESTAMP) <= days))
                 .groupBy(df_base.BUYER_ID, df_base.ITEM_ID, df_base.TITLE, df_base.EVENT_TIMESTAMP)
                 .agg(F.collect_list(df_action.TITLE).alias("hist_ttls"))
                 .withColumn("hist_ttls_cosine", fe.cosine_sim_to_target(df_base.TITLE, F.col("hist_ttls")))
                 .withColumn("hist_ttls_jaccard", fe.jaccard_sim_to_target(df_base.TITLE, F.col("hist_ttls")))
                 .drop("hist_ttls").drop(df_base.TITLE)
                 
     )
    return df_res


def f_cat_propensity_sim(df_base, df_action, action_type="purchase", cat_type="LEAF", days=1):
    df_res = _cat_sim_to_hist_actions(df_base, df_action, cat_type, days)
    # calculate aggregated features
    df_res = (df_res.withColumn(f"f_max_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", F.array_max(df_res.hist_cats_cosine))
                   .withColumn(f"f_max_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", F.array_max(df_res.hist_cats_jaccard))
                   .withColumn(f"f_min_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", F.array_min(df_res.hist_cats_cosine))
                   .withColumn(f"f_min_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", F.array_min(df_res.hist_cats_jaccard))
                   .withColumn(f"f_mean_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", fe.array_mean(df_res.hist_cats_cosine))
                   .withColumn(f"f_mean_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", fe.array_mean(df_res.hist_cats_jaccard))
                   .withColumn(f"f_var_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", fe.array_var(df_res.hist_cats_cosine))
                   .withColumn(f"f_var_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", fe.array_var(df_res.hist_cats_jaccard))
                   .withColumn(f"f_p25_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_cats_cosine, F.lit(25)))
                   .withColumn(f"f_p25_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_cats_jaccard, F.lit(25)))
                   .withColumn(f"f_p50_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_cats_cosine, F.lit(50)))
                   .withColumn(f"f_p50_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_cats_jaccard, F.lit(50)))
                   .withColumn(f"f_p75_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_cats_cosine, F.lit(75)))
                   .withColumn(f"f_p75_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_cats_jaccard, F.lit(75)))
                   .withColumn(f"f_scoreratio_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.score_ratio(F.col(f"f_min_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days"),
                               F.col(f"f_max_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days")))
                    .withColumn(f"f_scoreratio_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.score_ratio(F.col(f"f_min_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days"),
                               F.col(f"f_max_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days")))
                    .withColumn(f"f_poisson_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                                fe.poisson_likelihood(F.col(f"f_mean_{cat_type}_cat_{action_type}_propensity_cosine_sim_in_last_{days}_days")))
                    .withColumn(f"f_poisson_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                                fe.poisson_likelihood(F.col(f"f_mean_{cat_type}_cat_{action_type}_propensity_jaccard_sim_in_last_{days}_days")))
                    .drop(df_res.hist_cats_cosine)
                    .drop(df_res.hist_cats_jaccard)
             )
    return df_res


def f_title_propensity_sim(df_base, df_action, action_type="purchase", days=1):
    df_res = _title_sim_to_hist_actions(df_base, df_action, days)
    # calculate aggregated features
    df_res = (df_res.withColumn(f"f_max_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", F.array_max(df_res.hist_ttls_cosine))
                   .withColumn(f"f_max_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", F.array_max(df_res.hist_ttls_jaccard))
                   .withColumn(f"f_min_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", F.array_min(df_res.hist_ttls_cosine))
                   .withColumn(f"f_min_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", F.array_min(df_res.hist_ttls_jaccard))
                   .withColumn(f"f_mean_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", fe.array_mean(df_res.hist_ttls_cosine))
                   .withColumn(f"f_mean_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", fe.array_mean(df_res.hist_ttls_jaccard))
                   .withColumn(f"f_var_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", fe.array_var(df_res.hist_ttls_cosine))
                   .withColumn(f"f_var_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", fe.array_var(df_res.hist_ttls_jaccard))
                   .withColumn(f"f_p25_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_ttls_cosine, F.lit(25)))
                   .withColumn(f"f_p25_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_ttls_jaccard, F.lit(25)))
                   .withColumn(f"f_p50_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_ttls_cosine, F.lit(50)))
                   .withColumn(f"f_p50_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_ttls_jaccard, F.lit(50)))
                   .withColumn(f"f_p75_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_ttls_cosine, F.lit(75)))
                   .withColumn(f"f_p75_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.array_percentile(df_res.hist_ttls_jaccard, F.lit(75)))
                   .withColumn(f"f_scoreratio_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                               fe.score_ratio(F.col(f"f_min_title_{action_type}_propensity_cosine_sim_in_last_{days}_days"),
                               F.col(f"f_max_title_{action_type}_propensity_cosine_sim_in_last_{days}_days")))
                    .withColumn(f"f_scoreratio_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                               fe.score_ratio(F.col(f"f_min_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days"),
                               F.col(f"f_max_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days")))
                    .withColumn(f"f_poisson_title_{action_type}_propensity_cosine_sim_in_last_{days}_days", 
                                fe.poisson_likelihood(F.col(f"f_mean_title_{action_type}_propensity_cosine_sim_in_last_{days}_days")))
                    .withColumn(f"f_poisson_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days", 
                                fe.poisson_likelihood(F.col(f"f_mean_title_{action_type}_propensity_jaccard_sim_in_last_{days}_days")))
                    .drop(df_res.hist_ttls_cosine)
                    .drop(df_res.hist_ttls_jaccard)
             )
    return df_res


def vi_label(session, config, df_ftrs, df_vi):
    df_pos_vis = (df_vi.where(df_vi.EVENT_TIMESTAMP > config['sampling']['ref_dt'])
                  .select("BUYER_ID", "LEAF_CATEG_ID").dropDuplicates()
                  .withColumn("vi_label", F.lit(1)))
    df = (df_ftrs.join(df_pos_vis, ["BUYER_ID", "LEAF_CATEG_ID"], how="left")
                 .fillna(0))
    return df
         



                   
     
    