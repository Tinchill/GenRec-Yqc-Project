import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# --- 1. 初始化 ---
# 这里的 Key 会从 Streamlit Cloud 的 Secrets 中读取
api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

@st.cache_resource
def load_model():
    # 多语言语义模型，负责中英文对齐
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    try:
        # 仅读取前 10,000 行，平衡内存与覆盖率
        products = pd.read_csv("amazon_products.csv", nrows=10000)
        categories = pd.read_csv("amazon_categories.csv")
        
        cat_dict = dict(zip(categories['id'], categories['category_name']))
        products['category_name'] = products['category_id'].map(cat_dict).fillna("Other")
        
        # 预清洗数值列
        products['price'] = pd.to_numeric(products['price'], errors='coerce').fillna(0.0)
        products['stars'] = pd.to_numeric(products['stars'], errors='coerce').fillna(0.0)
        products['boughtInLastMonth'] = pd.to_numeric(products['boughtInLastMonth'], errors='coerce').fillna(0)
        products['reviews'] = pd.to_numeric(products['reviews'], errors='coerce').fillna(0)
        
        # 构造检索文本
        products['search_text'] = (products['category_name'] + " " + products['title'].fillna("")).str.lower()
        return products
    except Exception as e:
        st.error(f"数据加载异常: {e}")
        return pd.DataFrame()

# --- 2. 关键词转换助手 ---
def translate_query_to_keywords(query):
    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"Translate this product search to 3 core English keywords: '{query}'. Only return keywords."}]
        )
        return res.choices.message.content.strip()
    except:
        return ""

# --- 3. 推荐逻辑 ---
def get_recommendations(query, df, model, sort_mode, top_k=3):
    # 1. 尝试关键词初步缩小范围
    eng_keywords = translate_query_to_keywords(query)
    kw_list = [k.strip() for k in re.split(r'[, ]+', eng_keywords) if len(k) > 1]
    
    if kw_list:
        pattern = '|'.join(kw_list)
        candidates = df[df['search_text'].str.contains(pattern, case=False, na=False)].copy()
    else:
        candidates = pd.DataFrame()

    # 2. 如果关键词没中，或者太少，则直接用前 500 条做语义对比
    if len(candidates) < 5:
        candidates = df.head(500).copy()

    # 3. 语义向量计算
    query_vec = model.encode(query, convert_to_tensor=True)
    doc_vectors = model.encode(candidates['search_text'].tolist(), convert_to_tensor=True)
    cos_sims = util.cos_sim(query_vec, doc_vectors).cpu().numpy().flatten()
    candidates['sim_score'] = cos_sims

    # 4. 判定是否有相关结果 (设定相似度阈值)
    if candidates['sim_score'].max() < 0.2:
        return pd.DataFrame() # 返回空表表示没找到

    # 5. 综合业务排序
    if sort_mode == "价格优先":
        rank_score = 1 - (candidates['price'] / (candidates['price'].max() + 1))
    elif sort_mode == "评分优先":
        rank_score = candidates['stars'] / 5.0
    else:
        rank_score = candidates['boughtInLastMonth'] / (candidates['boughtInLastMonth'].max() + 1)

    candidates['final_score'] = candidates['sim_score'] * 0.7 + rank_score * 0.3
    return candidates.sort_values(by='final_score', ascending=False).head(top_k)

# --- 4. UI 界面 ---
st.set_page_config(page_title="Amazon Pro-GenRec", layout="wide")
df_all = load_data()
model_engine = load_model()

st.title("🛒 Amazon 商品智能助手")

with st.sidebar:
    st.header("🎯 搜索设置")
    sort_choice = st.radio("排序权重:", ["销量优先", "价格优先", "评分优先"])
    rec_limit = st.slider("推荐条数", 1, 5, 3)
    st.divider()
    st.caption(f"当前索引范围: 前 {len(df_all)} 条数据")

query = st.text_input("🔍 您想买什么？", placeholder="例如：给5岁男孩的蓝色乐高玩具")

if query:
    with st.spinner("正在为您检索最匹配的商品..."):
        results = get_recommendations(query, df_all, model_engine, sort_choice, rec_limit)
    
    if results.empty:
        st.error("💡 抱歉，在库中没有找到与您描述高度匹配的商品。建议换个关键词试试！")
    else:
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    st.image(row['imgUrl'], use_container_width=True)
                with col2:
                    st.subheader("📦 商品规格")
                    st.write(f"**名称**: {row['title']}")
                    st.write(f"💰 **价格**: `${row['price']}`")
                    st.write(f"⭐ **评分**: `{row['stars']}` ({int(row['reviews'])}条评论)")
                    st.write(f"🔥 **上月销量**: `{int(row['boughtInLastMonth'])}`")
                    st.caption(f"分类: {row['category_name']}")
                with col3:
                    st.subheader("🤖 AI 导购点评")
                    try:
                        res = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": f"用户需求:{query}\n推荐商品:{row['title']}\n请点评该商品与需求的契合度。"}]
                        )
                        st.info(res.choices.message.content)
                    except:
                        st.write("DeepSeek AI 正在休息，请稍后再试。")

st.divider()
st.caption("注：本系统仅检索前 10,000 条样本数据以保证响应速度。")
