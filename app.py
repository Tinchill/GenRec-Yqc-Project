import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# --- 1. 配置与 API 初始化 ---
if "DEEPSEEK_API_KEY" in st.secrets:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
else:
    api_key = "你的_API_KEY"

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# --- 2. 数据加载与清洗 (双表关联) ---
@st.cache_data
def load_and_merge_data():
    # 读取产品表
    products = pd.read_csv("amazon_products.csv")
    # 读取类目表 (假设列名为 id, category_name)
    categories = pd.read_csv("amazon_categories.csv")
    
    # 建立映射关系
    cat_dict = dict(zip(categories['id'], categories['category_name']))
    products['category_name'] = products['category_id'].map(cat_dict).fillna("Other")
    
    # 清洗数值列
    products['price'] = pd.to_numeric(products['price'], errors='coerce').fillna(0.0)
    products['stars'] = pd.to_numeric(products['stars'], errors='coerce').fillna(0.0)
    products['boughtInLastMonth'] = pd.to_numeric(products['boughtInLastMonth'], errors='coerce').fillna(0)
    
    # 构造语义特征：标题 + 类目 (这是强关联的关键)
    products['combined_info'] = products['category_name'] + " " + products['title'].fillna("")
    
    return products, categories

# --- 3. 增强版推荐引擎 ---
class ProGenRecSystem:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        # 预计算全量语义向量
        with st.spinner("系统正在构建多模态语义对齐索引..."):
            self.vectors = self.model.encode(self.df['combined_info'].tolist(), convert_to_tensor=True)

    def get_recommendations(self, query, sort_mode, top_k=3):
        # A. 语义初筛 (Semantic Retrieval)
        query_vec = self.model.encode(query, convert_to_tensor=True)
        cos_sims = util.cos_sim(query_vec, self.vectors).cpu().numpy().flatten()
        
        # 设定阈值 0.3，过滤掉完全不相关的品类 
        relevant_indices = np.where(cos_sims > 0.3)[0]
        if len(relevant_indices) == 0:
            relevant_indices = np.argsort(cos_sims)[-20:] # 兜底策略
            
        candidate_df = self.df.iloc[relevant_indices].copy()
        candidate_sims = cos_sims[relevant_indices]

        # B. 多维度重排 (Re-ranking)
        # 将语义相关性作为基础分 (占 50%)
        # 排序因子归一化
        if sort_mode == "价格优先":
            # 价格越低越好，进行反向归一化
            rank_score = 1 - (candidate_df['price'] / (candidate_df['price'].max() + 1))
        elif sort_mode == "评分优先":
            rank_score = candidate_df['stars'] / 5.0
        else: # 销量优先
            rank_score = candidate_df['boughtInLastMonth'] / (candidate_df['boughtInLastMonth'].max() + 1)

        # 最终得分 = 语义对齐(60%) + 业务排序(40%)
        candidate_df['final_score'] = candidate_sims * 0.6 + rank_score.values * 0.4
        
        return candidate_df.sort_values(by='final_score', ascending=False).head(top_k)

# --- 4. Streamlit UI 交互界面 ---
st.set_page_config(page_title="Amazon Pro-GenRec", layout="wide")

@st.cache_resource
def init_models():
    # 使用针对多模态描述优化的模型
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 数据加载
df_all, df_cats = load_and_merge_data()
model_engine = init_models()
recommender = ProGenRecSystem(df_all, model_engine)

st.title("🛒 Amazon 生成式推荐系统 (专业增强版)")
st.markdown("基于 **DeepSeek-V3** 与 **语义对齐技术**，实现精准的品类、颜色及属性匹配。")

# 侧边栏配置
with st.sidebar:
    st.header("🎯 推荐策略")
    sort_choice = st.radio("排序维度:", ["销量优先", "价格优先", "评分优先"])
    rec_limit = st.slider("推荐商品数量", 1, 4, 3)
    st.divider()
    st.write(f"📊 已关联类目数: {len(df_cats)}")
    if st.button("导出标准化 JSON"):
        st.json(df_all.head(50).to_dict(orient='records'))

# 交互输入
query = st.text_input("💬 输入需求 (支持颜色/尺寸/性别描述):", placeholder="例如：蓝色 男款 大号 运动鞋")

if query:
    results = recommender.get_recommendations(query, sort_choice, rec_limit)
    
    st.subheader(f"✨ 按照【{sort_choice}】为您推荐以下商品：")
    
    for idx, row in results.iterrows():
        with st.container(border=True):
            col_img, col_info, col_ai = st.columns([1.5, 3, 4])
            
            with col_img:
                st.image(row['imgUrl'], use_container_width=True)
            
            with col_info:
                st.markdown(f"### {row['title'][:80]}...")
                st.write(f"🏷️ **类目**: {row['category_name']}")
                st.write(f"💰 **价格**: ${row['price']}")
                st.write(f"⭐ **评分**: {row['stars']}")
                st.write(f"🔥 **上月销量**: {int(row['boughtInLastMonth'])}")
            
            with col_ai:
                with st.spinner("DeepSeek 正在解析属性对齐度..."):
                    # 构造大模型 Prompt：强调属性匹配
                    prompt = f"""
                    你是一个资深亚马逊导购。用户需求是“{query}”。
                    你选中的商品标题是“{row['title']}”，类目是“{row['category_name']}”。
                    请从【品类匹配度】、【属性对齐（颜色/性别/尺寸）】、【购买建议】三个维度生成思维链(CoT)推理。
                    要求：如果商品信息与需求不匹配，请诚实指出。
                    """
                    try:
                        res = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(res.choices.message.content)
                    except:
                        st.error("DeepSeek API 连接超时，请稍后重试。")

st.divider()
st.caption("AI for Rec Demo: 融合了语义相似度与硬性业务指标的生成式闭环。")
