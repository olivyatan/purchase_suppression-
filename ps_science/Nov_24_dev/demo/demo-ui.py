import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout="wide")

st.title(":no_entry_sign: Purchase Suppression Demo")

simple_feature_list = [
 'f_num_views_from_LEAF_cat_in_last_30_days',
 'f_num_views_from_LEAF_cat_in_last_14_days',
 'f_num_purchases_from_LEAF_cat_in_last_60_days',
 'f_num_views_from_LEAF_cat_in_last_60_days',
 'f_num_purchases_from_LVL2_cat_in_last_60_days',
 'f_max_LEAF_cat_view_propensity_cosine_sim_in_last_30_days',
 'f_time_since_last_view_from_LEAF_cat',
 'f_num_views_from_LEAF_cat_in_last_2_days',
 'f_num_purchases_from_LEAF_cat_in_last_30_days',
 'f_max_META_cat_view_propensity_jaccard_sim_in_last_60_days',
 'f_num_views_from_LEAF_cat_in_last_7_days',
 'f_num_views_from_LEAF_cat_in_last_5_days',
 'f_max_LEAF_cat_view_propensity_jaccard_sim_in_last_30_days',
 'f_time_since_last_view_from_LVL2_cat',
 'f_max_LEAF_cat_view_propensity_cosine_sim_in_last_14_days',
 'f_max_LEAF_cat_view_propensity_cosine_sim_in_last_60_days',
 'f_num_purchases_from_LEAF_cat_in_last_14_days',
 'f_max_LEAF_cat_view_propensity_jaccard_sim_in_last_60_days',
 'f_max_LVL2_cat_view_propensity_jaccard_sim_in_last_60_days',
 'f_max_LVL2_cat_purchase_propensity_jaccard_sim_in_last_30_days',
 'f_num_views_from_LVL2_cat_in_last_60_days',
 'f_num_views_from_LVL2_cat_in_last_5_days',
 'f_time_since_last_purchase_from_LEAF_cat',
 'f_max_LEAF_cat_view_propensity_cosine_sim_in_last_7_days',
 'f_max_LVL2_cat_view_propensity_cosine_sim_in_last_60_days',
 'f_min_LEAF_cat_view_propensity_cosine_sim_in_last_60_days',
 'f_max_LVL2_cat_purchase_propensity_jaccard_sim_in_last_60_days',
 'f_max_LEAF_cat_purchase_propensity_cosine_sim_in_last_30_days',
 'f_max_LVL2_cat_purchase_propensity_cosine_sim_in_last_30_days',
 'f_max_LEAF_cat_view_propensity_jaccard_sim_in_last_14_days'
]



def load_data():
    df_users = pd.read_csv("demo_users.cvs")
    df_vis = pd.read_csv("demo_vis.cvs")
    df_vis["Action"] = "view"
    df_vis["LEAF_CATEG_NAME"] = df_vis["LEAF_CATEG_NAME"].apply(lambda x: x.replace(':', ' / '))
    df_prchs = pd.read_csv("demo_prchs.cvs")
    df_prchs["Action"] = "purchase"
    df_prchs["LEAF_CATEG_NAME"] = df_prchs["LEAF_CATEG_NAME"].apply(lambda x: x.replace(':', ' / '))
    st.session_state["users"] = df_vis["BUYER_ID"].drop_duplicates().to_list()
    return df_users, df_vis, df_prchs

def classify_users(df_users, model_variant="simplified"):
    if model_variant == "simplified": 
        xgb_model = pickle.load(open("xgb_model_60.pkl", "rb"))
        feature_list = simple_feature_list
    else:
        xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
        feature_list = [f for f in df_users.columns if f.startswith("f_")]
    df_users["pred"] = xgb_model.predict(df_users[feature_list])
    return df_users

def check_prediction(category, future_cats, prediction):
    if category in future_cats and prediction == 1: 
        return True
    elif category not in future_cats and prediction == 0:
        return True
    return False
    
    
df_users, df_vis, df_prchs = load_data()
model_var_col,user_id_col,_ = st.columns([1,1,3])
model_variant = model_var_col.selectbox(label="Model variant", options=["simplified", "full"])
df_users = classify_users(df_users, model_variant)
user_id = user_id_col.selectbox(label="Select user", options=st.session_state["users"])
user_vi_hist = df_vis[df_vis["BUYER_ID"] == user_id][df_vis["EVENT_TIMESTAMP"] < '2024-05-01']
user_future_vis = df_vis[df_vis["BUYER_ID"] == user_id][df_vis["EVENT_TIMESTAMP"] > '2024-05-01']
user_prch_hist = df_prchs[df_prchs["BUYER_ID"] == user_id][df_prchs["EVENT_TIMESTAMP"] < '2024-05-01']
user_future_prchs = df_prchs[df_prchs["BUYER_ID"] == user_id][df_prchs["EVENT_TIMESTAMP"] > '2024-05-01']

user_history = pd.concat([user_vi_hist,user_prch_hist])
user_future = pd.concat([user_future_vis,user_future_prchs])

user_history = user_history[["Action", "EVENT_TIMESTAMP", "gallery_url", "LEAF_CATEG_NAME", "ITEM_ID", "LEAF_CATEG_ID"]].sort_values("EVENT_TIMESTAMP", ascending=False)

user_features = df_users[df_users["BUYER_ID"] == user_id]
future_cats = user_future["LEAF_CATEG_ID"].values
user_future = user_future[["Action", "EVENT_TIMESTAMP", "gallery_url", "LEAF_CATEG_NAME", "ITEM_ID", "LEAF_CATEG_ID"]].sort_values("EVENT_TIMESTAMP", ascending=False)
category = user_features["LEAF_CATEG_NAME"].values[0]
cat_id = user_features["LEAF_CATEG_ID"].values[0]
prediction = user_features["pred"].values[0]
check = check_prediction(cat_id, future_cats, prediction)


def green_color(row):
    return ['background-color:green'] * len(row) if row.LEAF_CATEG_ID == cat_id else ['background-color:white'] * len(row)

def red_color(row):
    return ['background-color:red'] * len(row) if row.LEAF_CATEG_ID == cat_id else ['background-color:white'] * len(row)
    

st.subheader("Prediction")
st.markdown(f"**Leaf Category**: ({cat_id}) {category.replace(':', ' / ')}")
st.markdown(f"**Prediction**: {'Suppress' if prediction == 0 else 'Recommend'} {':white_check_mark:' if check else ':x:'}")



st.subheader("Next 2 months actions")

only_rel_f_events = st.checkbox("Show only relevant future actions", value=True)
if only_rel_f_events:
    user_future = user_future[user_future["LEAF_CATEG_ID"] == cat_id]
if check:
    user_future = user_future.style.apply(green_color, axis=1)   
else:
    user_future = user_future.style.apply(red_color, axis=1)
    
st.data_editor(
    user_future,
    column_config={
        "gallery_url": st.column_config.ImageColumn(
            "Image", help="Item image",
        ),
        "ITEM_ID": st.column_config.NumberColumn(
            format="%d"
        ),
        "LEAF_CATEG_ID": st.column_config.NumberColumn(
            format="%d"
        )
    },
    hide_index=True,
)

st.subheader("User history")
only_rel_h_events = st.checkbox("Show only relevant history actions", value=True)
if only_rel_h_events:
    user_history = user_history[(user_history["LEAF_CATEG_ID"] == cat_id) & (user_history["Action"] == "purchase")]
if check:
    user_history = user_history.style.apply(green_color, axis=1)
else:
    user_history = user_history.style.apply(red_color, axis=1)
    
st.data_editor(
    user_history,
    column_config={
        "gallery_url": st.column_config.ImageColumn(
            "Image", help="Item image",
        ),
        "ITEM_ID": st.column_config.NumberColumn(
            format="%d"
        ),
        "LEAF_CATEG_ID": st.column_config.NumberColumn(
            format="%d"
        )
    },
    hide_index=True,
)

st.subheader("Features")
st.dataframe(user_features)

    