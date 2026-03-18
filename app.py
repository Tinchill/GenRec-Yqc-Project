import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# --- 1. 初始化 ---
api_key = st.secrets.get("DEEPSEEK_API_KEY", "111111")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

@st.cache_resource
def load_model():
    # 多语言语义模型
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    products = pd.read_csv("amazon_products.csv")
    categories = pd.read_csv("amazon_categories.csv")
    cat_dict = dict(zip(categories['id'], categories['category_name']))
    products['category_name'] = products['category_id'].map(cat_dict).fillna("Other")
    
    # 基础清洗
    products['price'] = pd.to_numeric(products['price'], errors='coerce').fillna(0.0)
    products['stars'] = pd.to_numeric(products['stars'], errors='coerce').fillna(0.0)
    products['reviews'] = pd.to_numeric(products['reviews'], errors='coerce').fillna(0)
    products['boughtInLastMonth'] = pd.to_numeric(products['boughtInLastMonth'], errors='coerce').fillna(0)
    products['search_text'] = (products['category_name'] + " " + products['title'].fillna("")).str.lower()
    return products

# --- 2. 动态提取关键词 (DeepSeek 助力) ---
def get_keywords_from_ai(query):
    prompt = f"Extract 3-5 essential English keywords for Amazon product search from this user request: '{query}'. Return only keywords separated by space."
    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices.message.content.strip()
    except:
        return query # 失败则回退

# --- 3. 动态过滤与推荐逻辑 ---
def get_recommendations(query, df, model, sort_mode, top_k=3):
    # 第一步：获取英文关键词
    eng_keywords = get_keywords_from_ai(query)
    kw_list = [k.strip() for k in re.split(r'[, ]+', eng_keywords) if len(k) > 2]
    
    # 第二步：全量表快速初筛 (利用正则表达式，不限行数)
    pattern = '|'.join(kw_list)
    # 找标题或类目中包含任何一个关键词的行
    candidates = df[df['search_text'].str.contains(pattern, case=False, na=False)].copy()
    
    # 如果没搜到，保底取前 300 条相关度最高的
    if len(candidates) < 10:
        candidates = df.head(300).copy()
    else:
        # 如果搜到太多，取前 300 条进行精排（保证内存安全）
        candidates = candidates.head(300)

    # 第三步：对筛选出的 300 条进行深度语义计算
    query_vec = model.encode(query, convert_to_tensor=True)
    doc_vectors = model.encode(candidates['search_text'].tolist(), convert_to_tensor=True)
    cos_sims = util.cos_sim(query_vec, doc_vectors).cpu().numpy().flatten()
    candidates['sim_score'] = cos_sims

    # 第四步：结合业务权重排序
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

st.title("🛒 Amazon 动态语义推荐系统")

with st.sidebar:
    st.header("⚙️ 排序策略")
    sort_choice = st.radio("权重偏好:", ["销量优先", "价格优先", "评分优先"])
    rec_limit = st.slider("呈现数量", 1, 5, 3)
    st.info(f"检索底库: {len(df_all)} 条")

query = st.text_input("💬 您想找什么？", placeholder="例如：透气舒适的 Nike 跑步鞋")

if query:
    with st.status("🚀 正在跨语言检索全量数据库...") as status:
        st.write("1. 正在通过 DeepSeek 翻译并提取关键词...")
        results = get_recommendations(query, df_all, model_engine, sort_choice, rec_limit)
        status.update(label="✅ 检索完成！", state="complete")

    for _, row in results.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.image(row['imgUrl'], use_container_width=True)
            with c2:
                st.subheader("📦 商品清单")
                st.markdown(f"**{row['title']}**")
                st.write(f"💰 价格: `${row['price']}` | ⭐ 评分: `{row['stars']}`")
                st.write(f"🔥 上月销量: `{int(row['boughtInLastMonth'])}` | 💬 评价数: `{int(row['reviews'])}`")
                st.caption(f"Category: {row['category_name']}")
            with c3:
                st.subheader("🤖 AI 点评")
                # 此处保留 DeepSeek 深度点评
                try:
                    res = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": f"用户需求:{query}\n商品:{row['title']}\n请给出推荐理由。"}]
                    )
                    st.info(res.choices.message.content)
                except:
                    st.write("AI 暂时无法评价")
