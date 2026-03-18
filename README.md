# Amazon-GenRec-Pro: 基于 DeepSeek 的专业级生成式推荐系统

![Streamlit App](https://static.streamlit.io) 
**核心架构：** 语义对齐召回 (Sentence-Transformer) + 业务逻辑重排 (Rerank) + 大模型推理 (DeepSeek-V3/R1)

---

## 🌟 项目亮点
这是一个模拟工业级“召回-精排-生成”链路的电商推荐演示系统。项目针对传统推荐系统“结果不可解释”和“品类偏移”的痛点，通过 **DeepSeek 大模型** 实现了具备 **思维链 (CoT)** 推理能力的生成式推荐。

- **强类目关联**：通过 `category_id` 实时映射类目文本，确保推荐结果不跑偏。
- **细粒度属性对齐**：系统能精准解析用户指令中的 **颜色、性别、尺寸、材质** 等信息（从商品 Title 中提取）。
- **三维度智能排序**：支持 **价格优先、评分优先、销量优先** 三种业务重排逻辑。
- **透明化推理 (CoT)**：调用 DeepSeek API，实时生成推荐理由并校验商品属性匹配度。

---

## 🚀 快速运行指南

### 1. 环境准备
确保您的 Python 版本 $\ge$ 3.9，克隆仓库后安装依赖：
```bash
git clone https://github.com
cd GenRec-Project
pip install -r requirements.txt
