import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# --- 1. 配置与 API 初始化 ---
# 优先从 Secrets 读取，本地调试可回退
api_key = st.secrets.get("DEEPSEEK_API_KEY", "YOUR_BACKUP_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# --- 2. 数据加载 (增加数据量限制防止内存溢出) ---
@st.cache_data
def load_and_merge_data(nrows=500): # 默认先加载500条测试，稳定后再加大
    try:
        products = pd.read_csv("amazon_products.csv", nrows=nrows)
        categories = pd.read_csv("amazon_categories.csv")
        
        cat_dict = dict(zip(categories['id'], categories['category_name']))
        products['category_name'] = products['category_id'].map(cat_dict).fillna("Other")
        
        # 清洗数值
        for col in ['price', 'stars', 'boughtInLastMonth']:
            products[col] = pd.to_numeric(products[col], errors='coerce').fillna(0)
            
        products['combined_info'] = products['category_name'] + " " + products['title'].fillna("")
        return products, categories
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. 内存友好型推荐引擎 ---
@st.cache_resource
def load_model():
    # 使用超轻量模型，减少内存占用
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_recommendations(query, df, model, sort_mode, top_k=3):
    # 步骤 A: 简单的关键词预筛选 (减少后续向量计算量)
    # 如果数据量大，这里先搜素包含 query 关键词的行，或者随机抽样
    sample_df = df.head(500).copy() 
    
    # 步骤 B: 语义计算
    query_vec = model.encode(query, convert_to_tensor=True)
    doc_vectors = model.encode(sample_df['combined_info'].tolist(), convert_to_tensor=True)
    cos_sims = util.cos_sim(query_vec, doc_vectors).cpu().numpy().flatten()
    
    sample_df['sim_score'] = cos_sims

    # 步骤 C: 业务排序权重
    if sort_mode == "价格优先":
        rank_score = 1 - (sample_df['price'] / (sample_df['price'].max() + 1))
    elif sort_mode == "评分优先":
        rank_score = sample_df['stars'] / 5.0
    else:
        rank_score = sample_df['boughtInLastMonth'] / (sample_df['boughtInLastMonth'].max() + 1)

    sample_df['final_score'] = sample_df['sim_score'] * 0.6 + rank_score * 0.4
    return sample_df.sort_values(by='final_score', ascending=False).head(top_k)

# --- 4. UI 界面 ---
st.set_page_config(page_title="Amazon AI Rec", layout="wide")

# 加载数据和模型
df_all, df_cats = load_and_merge_data(nrows=1000) # 建议先从1000条开始
model_engine = load_model()

st.title("🛒 Amazon 生成式推荐 (轻量版)")

if df_all.empty:
    st.warning("请确保 CSV 文件已上传至 GitHub 仓库根目录。")
    st.stop()

with st.sidebar:
    st.header("🎯 设置")
    sort_choice = st.radio("排序维度:", ["销量优先", "价格优先", "评分优先"])
    rec_limit = st.slider("推荐数量", 1, 4, 3)
    st.info(f"当前索引数据量: {len(df_all)} 条")

query = st.text_input("💬 您想买什么？", placeholder="例如：红色 运动鞋")

if query:
    with st.spinner("正在匹配最佳商品..."):
        results = get_recommendations(query, df_all, model_engine, sort_choice, rec_limit)
    
    for _, row in results.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 3])
            with c1:
                st.image(row['imgUrl'])
            with c2:
                st.subheader(row['title'][:50] + "...")
                st.write(f"💰 ${row['price']} | ⭐ {row['stars']}")
                st.caption(f"类目: {row['category_name']}")
            with c3:
                # 调用 DeepSeek
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": f"用户想找{query}，请评价这款商品：{row['title']}"}]
                    )
                    st.markdown(resp.choices[0].message.content)
                except:
                    st.write("AI 评价暂时不可用")
