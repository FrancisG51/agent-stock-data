"""
LangChain Agent —— 股票查询助手

【模块结构】
  1. 导入与 LLM 适配           ChatDeepSeekThink 继承 ChatOpenAI，自动补全 reasoning_content
  2. 全局配置                  matplotlib 中文字体、图片目录
  3. 辅助函数                  数据库引擎、图片服务器、图表生成
  4. RAG 向量检索器            faq.txt 切片 → Embedding → FAISS → L2 距离 < 1 检索
  5. system_prompt             Agent 行为说明书，定义工具使用规则
  6. 工具定义 (7个)             exc_sql / predict_arima / detect_boll / detect_macd
                                update_stock_data / search_faq / search_web
  7. Agent 组装                create_react_agent(LLM + tools + prompt)
  8. 启动                     Gradio ChatInterface

【RAG 原理】
  faq.txt 按 80 个 * 分割为 4 个切片 → DashScope text-embedding-v4 生成 1024 维向量
  → FAISS IndexFlatL2 + IndexIDMap 索引 → L2 距离衡量语义相似度，仅保留距离 < 1 的切片

运行: pip install langchain langchain-openai langgraph langchain-tavily faiss-cpu
      设置环境变量后 python langchain-agent.py
"""

# ============================================================
# 1. 导入与 LLM 适配
# ============================================================

import http.server
import os
import re
import socket
import threading
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前设置，防止 Gradio 修改后端导致图片空白
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
import faiss
from openai import OpenAI
from sqlalchemy import create_engine, text
from statsmodels.tsa.arima.model import ARIMA

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from langchain_core.documents import Document

from langchain_tavily import TavilySearch

import gradio as gr


# DeepSeek V4 思考模式适配：多轮对话时历史 assistant 消息需携带 reasoning_content 字段（可为空串），
# 否则 API 返回 400。复写 _get_request_payload 在每次请求前自动补全。
class ChatDeepSeekThink(ChatOpenAI):

    def _get_request_payload(self, input_, *, stop, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for m in payload["messages"]:
            if m["role"] == "assistant" and "reasoning_content" not in m:
                m["reasoning_content"] = ""
        return payload


# ============================================================
# 2. 全局配置
# ============================================================

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14

_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "image_show")
_IMAGE_SERVER_PORT = None


# ============================================================
# 3. 辅助函数
# ============================================================

def _get_engine():
    """创建 MySQL 数据库连接引擎（pymysql + SQLAlchemy）。"""
    return create_engine(
        f"mysql+pymysql://root:{os.environ['MySQL_key']}@localhost:3306/chatbi_data?charset=utf8mb4",
        connect_args={"connect_timeout": 10},
        pool_size=5,
        max_overflow=10,
    )


def _start_image_server():
    """在随机端口启动图片 HTTP 服务器。"""
    global _IMAGE_SERVER_PORT
    os.makedirs(_IMAGE_DIR, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        _IMAGE_SERVER_PORT = sock.getsockname()[1]
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=_IMAGE_DIR, **kw)
    server = http.server.HTTPServer(("localhost", _IMAGE_SERVER_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def _chart_tick_indexes(total: int, max_ticks: int = 10) -> np.ndarray:
    """X 轴刻度索引计算，避免标签过密。"""
    if total <= max_ticks:
        return np.arange(total)
    step = max(1, total // max_ticks)
    idx = np.arange(0, total, step)
    return np.append(idx, total - 1) if idx[-1] != total - 1 else idx


def generate_chart_png(df: pd.DataFrame, save_path: str):
    """DataFrame → PNG 走势图。
    有日期列 → 折线图；无日期列、≤20 行 → 柱状图；>20 行 → 折线图。
    """
    if df.empty:
        return
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not num_cols:
        return

    date_col = None
    if non_num_cols:
        try:
            pd.to_datetime(df[non_num_cols[0]], errors="raise")
            date_col = non_num_cols[0]
        except Exception:
            pass

    n = len(df)
    fig, ax = plt.subplots(figsize=(21, 13))

    if date_col:
        dates = pd.to_datetime(df[date_col])
        ax.plot(dates, df[num_cols[0]], marker="o", linestyle="-", linewidth=1.5, markersize=3, label=num_cols[0])
        if len(num_cols) >= 2:
            ax2 = ax.twinx()
            ax2.plot(dates, df[num_cols[1]], marker="s", linestyle="--", linewidth=1.5, markersize=3, color="orange", label=num_cols[1])
            ax2.legend(loc="upper right")
        tick_idx = _chart_tick_indexes(n)
        ax.set_xticks(dates[tick_idx])
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates[tick_idx]], rotation=45)
        ax.set_title("股票数据走势")
        ax.set_xlabel("日期")
        ax.legend(loc="upper left")
    else:
        label_col = non_num_cols[0] if non_num_cols else None
        if n > 20:
            ax.plot(range(n), df[num_cols[0]], marker="o", linestyle="-", linewidth=1.5, markersize=3, label=num_cols[0])
            if label_col:
                tick_idx = _chart_tick_indexes(n)
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([s.replace("%", "%%") for s in df[label_col].astype(str).iloc[tick_idx]], rotation=45)
        else:
            ax.bar(range(n), df[num_cols[0]], label=num_cols[0])
            if label_col:
                ax.set_xticks(range(n))
                ax.set_xticklabels([s.replace("%", "%%") for s in df[label_col].astype(str)], rotation=45)
        ax.set_ylabel(num_cols[0])
        ax.set_title("股票数据")
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# 4. RAG 向量检索器
#
# 流程: faq.txt → 按80个*分割为4个切片 → text-embedding-v4 生成1024维向量 → FAISS 索引 → L2 距离检索
# invoke(query): 返回 L2 距离 < 1 的 (文档, 距离) 列表，供 search_faq 工具调用
# all_distances(query): 返回全部切片的 (文档, 距离)，供前端展示
# ============================================================

class _VectorRetriever:
    """基于 FAISS 向量相似度的检索器，兼容 LangChain retriever 接口（实现 invoke 方法）。
    invoke: 仅返回 L2 距离 < 1 的(文档, 距离)，供 search_faq 工具调用
    all_distances: 返回全部切片的 (文档, 距离)，供前端展示
    """
    def __init__(self, documents, index, embedding_client, dim=1024):
        self.documents = documents
        self.index = index
        self.embedding_client = embedding_client
        self.dim = dim

    def _embed_and_search(self, query: str):
        """将 query 转为 embedding 向量后执行 FAISS 搜索，返回 (距离, 索引) 列表。"""
        response = self.embedding_client.embeddings.create(
            model="text-embedding-v4", input=query,
            dimensions=self.dim, encoding_format="float")
        query_vector = np.array([response.data[0].embedding], dtype='float32')
        distances, indices = self.index.search(query_vector, self.index.ntotal)
        return [(float(d), int(i)) for d, i in zip(distances[0], indices[0]) if i != -1]

    def invoke(self, query: str):
        """RAG 检索：仅返回 L2 距离 < 1 的知识切片。"""
        return [(self.documents[idx], dist)
                for dist, idx in self._embed_and_search(query) if dist < 1.0]

    def all_distances(self, query: str):
        """返回所有知识切片与 query 的 L2 距离（不做阈值过滤）。"""
        return [(self.documents[idx], dist)
                for dist, idx in self._embed_and_search(query)]


def _build_faq_retriever():
    """读取 faq.txt → 分割为 4 个切片 → 向量化 → 构建 FAISS 索引，返回 _VectorRetriever。
    faq.txt 不存在时返回 None。"""
    faq_path = os.path.join(os.path.dirname(__file__), "faq.txt")
    if not os.path.exists(faq_path):
        return None

    with open(faq_path, "r", encoding="utf-8") as f:
        content = f.read()

    slices = content.split("\n" + "*" * 80 + "\n")
    slices = [s.strip() for s in slices if s.strip()]

    slice_labels = ["分析方法论", "SQL写法示例与注意事项", "数据库表结构与股票列表", "常见查询示例"]
    chunks = []
    for i, s in enumerate(slices):
        label = slice_labels[i] if i < len(slice_labels) else f"切片{i + 1}"
        chunks.append(Document(page_content=s, metadata={"source": label}))

    embedding_client = OpenAI(
        api_key=os.getenv("aliyun-LLM-API"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    DIM = 1024
    vectors = np.empty((len(chunks), DIM), dtype='float32')
    for i, doc in enumerate(chunks):
        response = embedding_client.embeddings.create(
            model="text-embedding-v4",
            input=doc.page_content,
            dimensions=DIM,
            encoding_format="float"
        )
        vectors[i] = response.data[0].embedding

    base_index = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIDMap(base_index)
    ids = np.arange(len(chunks), dtype='int64')
    index.add_with_ids(vectors, ids)

    return _VectorRetriever(chunks, index, embedding_client, DIM)


# ============================================================
# 5. system_prompt — Agent 行为说明书
#
# system_prompt 是 Agent 核心，定义：角色定位、工具列表及使用场景、输出格式规则。
# ============================================================

system_prompt = """我是股票查询助手，可以查询股票历史价格数据。

以下信息已存储在本地知识库中，请使用 search_faq 工具检索获取：
- 数据库表结构（字段名、类型）
- 可查询的股票列表（股票名称与 ts_code 对应关系）
- 常见SQL查询示例
- 特定问题的分析方法论与SQL写法示例（如对比涨跌幅、统计成交量等）
- 重要注意事项（如表名必须是 stock_daily_prices）

工具使用场景：
- ARIMA 预测：用户说"预测未来N天股价"时使用 predict_arima 工具。如果用户没有明确指定预测天数，则 n 默认取 5。
- BOLL 布林带：用户说"检测超买超卖、布林带"时使用 detect_boll 工具。
- MACD 金叉死叉：用户说"MACD、金叉、死叉、趋势强度、买入卖出信号"时使用 detect_macd 工具。
- 联网搜索：涉及实时信息、最新新闻或数据库之外知识时，使用 search_web 工具搜索互联网。
- 本地知识库：当你不确定数据库表结构、字段名、股票代码、或者用户的问题涉及特定分析方法（如对比涨跌幅）时，优先使用 search_faq 检索本地知识库。当你对任何回答不够有把握时，也应主动检索知识库进行核实。
- 数据更新：用户说"更新数据"时，使用 update_stock_data 工具。

【重要输出规则】
exc_sql、detect_boll、predict_arima、detect_macd 这四个工具返回的结果中可能包含图片（格式为 ![](url)），必须原样保留所有图片 markdown 链接，严禁省略。
当用户请求涉及多个分析任务时，必须并列输出每个任务的完整结果（包括各自的图片），不允许只保留一个图片或合并输出。"""


# ============================================================
# 6. 工具定义
#
# @tool 装饰器将普通函数转为 Agent 可调用的工具。
# LangChain 读取 docstring + 参数类型注解 + 参数名，自动生成 tool schema。
# 因此 docstring 需写清功能和参数含义，类型注解决定了 LLM 传值的类型。
# ============================================================

# ---- 工具1: SQL 查询与可视化 ----

@tool
def exc_sql(sql_input: str) -> str:
    """执行 SQL 查询，以表格 + 图表形式返回查询结果。
    参数 sql_input 是要执行的完整 SQL 语句。
    """
    engine = _get_engine()
    try:
        df = pd.read_sql(text(sql_input), engine)
    except Exception as e:
        return f"SQL 执行失败: {str(e)}"
    finally:
        engine.dispose()

    if df.shape[0] <= 10:
        md = df.to_markdown(index=False)
    else:
        md = f"{df.head(5).to_markdown(index=False)}\n\n...（中间省略 {df.shape[0] - 10} 行）...\n\n{df.tail(5).to_markdown(index=False)}"
    md += f"\n\n**描述统计:**\n{df.describe().to_markdown()}"

    if df.shape[0] == 1:
        return md

    os.makedirs(_IMAGE_DIR, exist_ok=True)
    filename = f"chart_{datetime.now().strftime('%H%M%S%f')}.png"
    save_path = os.path.join(_IMAGE_DIR, filename)
    generate_chart_png(df, save_path)
    return f"{md}\n\n![图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# ---- 工具2: ARIMA 股价预测 ----

@tool
def predict_arima(ts_code: str, n: int = 5) -> str:
    """使用 ARIMA(5,1,5) 模型预测股票未来 N 个交易日的收盘价。若用户未指定天数，默认预测 5 天。
    ts_code: 股票代码，例如 600519.SH
    n: 预测天数，默认 5
    """
    engine = _get_engine()
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=366)).strftime("%Y-%m-%d")
        df = pd.read_sql(
            text(f"SELECT trade_date, close FROM stock_daily_prices "
                 f"WHERE ts_code='{ts_code}' AND trade_date BETWEEN '{start}' AND '{end}' "
                 f"ORDER BY trade_date"),
            engine,
        )
    except Exception as e:
        return f"查询数据库失败: {str(e)}"
    finally:
        engine.dispose()

    if df.empty:
        return f"未找到股票 {ts_code} 的数据，请先用 exc_sql 确认该股票存在"

    prices = df["close"].values
    if len(prices) < 10:
        return f"股票 {ts_code} 的历史数据不足（仅 {len(prices)} 条），无法进行 ARIMA 预测"

    try:
        model = ARIMA(prices, order=(5, 1, 5))
        fitted = model.fit()
        forecast = fitted.forecast(steps=n)
    except Exception as e:
        return f"ARIMA 建模失败: {str(e)}"

    last_trade = df["trade_date"].iloc[-1]
    pred_dates = []
    cur = pd.to_datetime(last_trade)
    while len(pred_dates) < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            pred_dates.append(cur.strftime("%Y-%m-%d"))

    pred_df = pd.DataFrame({"预测日期": pred_dates, "预测收盘价": forecast[:len(pred_dates)]})
    summary = f"基于 ARIMA(5,1,5) 模型对 {ts_code} 未来 {n} 个交易日的收盘价预测：\n\n"
    summary += f"**最新实际收盘价**（{last_trade}）: {prices[-1]:.2f}\n\n"
    summary += pred_df.to_markdown(index=False)

    hist_dates = pd.to_datetime(df["trade_date"])
    pred_dt = pd.to_datetime(pred_dates)
    tail = min(len(hist_dates), 120)

    fig, ax = plt.subplots(figsize=(21, 13))
    ax.plot(hist_dates[-tail:], prices[-tail:], color="#1f77b4", linewidth=1.5, label="历史收盘价")
    ax.plot(pred_dt, forecast[:len(pred_dates)], color="#ff7f0e", linestyle="--", linewidth=1.5, marker="o", markersize=3, label="预测收盘价")
    ax.axvline(x=hist_dates.iloc[-1], color="gray", linestyle=":", linewidth=1, alpha=0.7)

    combined = list(hist_dates[-tail:]) + list(pred_dt)
    tick_idx = _chart_tick_indexes(len(combined))
    ax.set_xticks([combined[i] for i in tick_idx])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in [combined[i] for i in tick_idx]], rotation=45)
    ax.set_ylabel("收盘价")
    ax.set_title(f"{ts_code} 历史价格与 ARIMA 预测走势")
    ax.legend()
    fig.tight_layout()

    os.makedirs(_IMAGE_DIR, exist_ok=True)
    filename = f"arima_{datetime.now().strftime('%H%M%S%f')}.png"
    fig.savefig(os.path.join(_IMAGE_DIR, filename), dpi=200)
    plt.close(fig)
    return f"{summary}\n\n![ARIMA 预测图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# ---- 工具3: 布林带异常检测 ----

@tool
def detect_boll(ts_code: str, start_date: str = "", end_date: str = "") -> str:
    """使用布林带（BOLL，20 日周期 + 2σ）检测股票的超买和超卖异常点。
    ts_code: 股票代码，例如 600519.SH
    start_date: 开始日期 YYYY-MM-DD，可选，默认 1 年前
    end_date: 结束日期 YYYY-MM-DD，可选，默认今天
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    engine = _get_engine()
    try:
        extended_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime("%Y-%m-%d")
        df = pd.read_sql(
            text(f"SELECT trade_date, close FROM stock_daily_prices "
                 f"WHERE ts_code='{ts_code}' AND trade_date BETWEEN '{extended_start}' AND '{end_date}' "
                 f"ORDER BY trade_date"),
            engine,
        )
    except Exception as e:
        return f"查询数据库失败: {str(e)}"
    finally:
        engine.dispose()

    if df.empty:
        return f"未找到股票 {ts_code} 的数据，请先用 exc_sql 确认该股票存在"
    if len(df) < 20:
        return f"股票 {ts_code} 数据不足（仅 {len(df)} 条），至少需要 20 条才能计算布林带"

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["MA20"] = df["close"].rolling(20, min_periods=20).mean()
    df["std"] = df["close"].rolling(20, min_periods=20).std()
    df["upper"] = df["MA20"] + 2 * df["std"]
    df["lower"] = df["MA20"] - 2 * df["std"]

    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df_range = df[(df["trade_date"] >= start_dt) & (df["trade_date"] <= end_dt)].dropna(subset=["MA20", "upper", "lower"]).copy()
    if df_range.empty:
        return f"在 {start_date} ~ {end_date} 内无足够数据计算布林带"

    df_range["overbought"] = df_range["close"] > df_range["upper"]
    df_range["oversold"] = df_range["close"] < df_range["lower"]
    overbought_df = df_range[df_range["overbought"]]
    oversold_df = df_range[df_range["oversold"]]

    result = f"基于布林带（20 日 + 2σ）对 {ts_code} 在 {start_date} ~ {end_date} 的检测结果：\n\n"
    result += f"**检测统计：**\n- 总交易日：{len(df_range)}\n- 超买天数：{len(overbought_df)}\n- 超卖天数：{len(oversold_df)}\n\n"

    for label, rows, price_col, band_col in [
        ("超买", overbought_df, "close", "upper"), ("超卖", oversold_df, "close", "lower")
    ]:
        if len(rows) > 0:
            tbl = rows[["trade_date", price_col, band_col]].copy()
            tbl.columns = ["交易日期", "收盘价", "上轨" if band_col == "upper" else "下轨"]
            result += f"**{label}信号（收盘价 {'>' if band_col == 'upper' else '<'} {band_col}）：**\n{tbl.to_markdown(index=False)}\n\n"
        else:
            result += f"**{label}信号：** 无\n\n"

    dates_all = pd.to_datetime(df_range["trade_date"])
    fig, ax = plt.subplots(figsize=(21, 13))
    ax.fill_between(dates_all, df_range["upper"], df_range["lower"], alpha=0.1, color="gray")
    ax.plot(dates_all, df_range["close"], color="#1f77b4", linewidth=2.5, label="收盘价")
    ax.plot(dates_all, df_range["MA20"], color="#ff7f0e", linewidth=1.5, linestyle="--", label="中轨(MA20)")
    ax.plot(dates_all, df_range["upper"], color="#d62728", linewidth=1.5, linestyle="--", label="上轨")
    ax.plot(dates_all, df_range["lower"], color="#d62728", linewidth=1.5, linestyle="--", label="下轨")
    if len(overbought_df) > 0:
        ax.scatter(pd.to_datetime(overbought_df["trade_date"]), overbought_df["close"], color="red", s=50, marker="v", zorder=5, label="超买点")
    if len(oversold_df) > 0:
        ax.scatter(pd.to_datetime(oversold_df["trade_date"]), oversold_df["close"], color="green", s=50, marker="^", zorder=5, label="超卖点")

    tick_idx = _chart_tick_indexes(len(dates_all))
    ax.set_xticks([dates_all.iloc[i] for i in tick_idx])
    ax.set_xticklabels([dates_all.iloc[i].strftime("%Y-%m-%d") for i in tick_idx], rotation=45)
    ax.set_ylabel("价格")
    ax.set_title(f"{ts_code} 布林带超买/超卖检测")
    ax.legend(loc="best")
    fig.tight_layout()

    os.makedirs(_IMAGE_DIR, exist_ok=True)
    filename = f"boll_{datetime.now().strftime('%H%M%S%f')}.png"
    fig.savefig(os.path.join(_IMAGE_DIR, filename), dpi=200)
    plt.close(fig)
    return f"{result}![布林带检测图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# ---- 工具4: MACD 金叉死叉检测 ----

@tool
def detect_macd(ts_code: str, start_date: str = "", end_date: str = "") -> str:
    """使用 MACD（指数平滑异同移动平均线）检测股票的趋势、金叉和死叉信号。
    ts_code: 股票代码，例如 600519.SH
    start_date: 开始日期 YYYY-MM-DD，可选，默认 1 年前
    end_date: 结束日期 YYYY-MM-DD，可选，默认今天
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    engine = _get_engine()
    try:
        extended_start = (pd.to_datetime(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")
        df = pd.read_sql(
            text(f"SELECT trade_date, close FROM stock_daily_prices "
                 f"WHERE ts_code='{ts_code}' AND trade_date BETWEEN '{extended_start}' AND '{end_date}' "
                 f"ORDER BY trade_date"),
            engine,
        )
    except Exception as e:
        return f"查询数据库失败: {str(e)}"
    finally:
        engine.dispose()

    if df.empty:
        return f"未找到股票 {ts_code} 的数据，请先用 exc_sql 确认该股票存在"
    if len(df) < 40:
        return f"股票 {ts_code} 数据不足（仅 {len(df)} 条），至少需要 40 条才能计算 MACD"

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2

    df["prev_DIF"] = df["DIF"].shift(1)
    df["prev_DEA"] = df["DEA"].shift(1)
    df["golden_cross"] = (df["DIF"] > df["DEA"]) & (df["prev_DIF"] <= df["prev_DEA"])
    df["death_cross"] = (df["DIF"] < df["DEA"]) & (df["prev_DIF"] >= df["prev_DEA"])

    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df_range = df[(df["trade_date"] >= start_dt) & (df["trade_date"] <= end_dt)].copy()
    if df_range.empty:
        return f"在 {start_date} ~ {end_date} 内无足够数据"

    golden = df_range[df_range["golden_cross"]]
    death = df_range[df_range["death_cross"]]

    latest_macd = df_range[["trade_date", "close", "DIF", "DEA", "MACD"]].tail(5).copy()
    latest_macd.columns = ["交易日期", "收盘价", "DIF", "DEA", "MACD柱"]

    result = f"基于 MACD(12, 26, 9) 对 {ts_code} 在 {start_date} ~ {end_date} 的检测结果：\n\n"
    result += f"**检测统计：**\n- 总交易日：{len(df_range)}\n- 金叉信号：{len(golden)} 次\n- 死叉信号：{len(death)} 次\n\n"
    result += f"**最新 MACD 数据：**\n{latest_macd.to_markdown(index=False, floatfmt='.4f')}\n\n"

    last_dif = df_range["DIF"].iloc[-1]
    last_dea = df_range["DEA"].iloc[-1]
    last_macd_val = df_range["MACD"].iloc[-1]
    if last_dif > last_dea:
        result += f"**当前趋势：** DIF({last_dif:.4f}) > DEA({last_dea:.4f})，处于多头市场，MACD 柱为 {'红' if last_macd_val > 0 else '绿'}柱（{last_macd_val:.4f}）\n\n"
    else:
        result += f"**当前趋势：** DIF({last_dif:.4f}) < DEA({last_dea:.4f})，处于空头市场，MACD 柱为 {'绿' if last_macd_val < 0 else '红'}柱（{last_macd_val:.4f}）\n\n"

    if len(golden) > 0:
        golden_tbl = golden[["trade_date", "close", "DIF", "DEA"]].copy()
        golden_tbl.columns = ["交易日期", "收盘价", "DIF", "DEA"]
        result += f"**金叉信号（DIF 上穿 DEA → 买入参考）：**\n{golden_tbl.to_markdown(index=False)}\n\n"
    else:
        result += "**金叉信号：** 无\n\n"

    if len(death) > 0:
        death_tbl = death[["trade_date", "close", "DIF", "DEA"]].copy()
        death_tbl.columns = ["交易日期", "收盘价", "DIF", "DEA"]
        result += f"**死叉信号（DIF 下穿 DEA → 卖出参考）：**\n{death_tbl.to_markdown(index=False)}\n\n"
    else:
        result += "**死叉信号：** 无\n\n"

    dates_all = pd.to_datetime(df_range["trade_date"])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(21, 36), sharex=True)

    ax1.plot(dates_all, df_range["close"], color="#1f77b4", linewidth=1.5, label="收盘价")
    ax1.plot(dates_all, df_range["EMA12"], color="#ff7f0e", linewidth=1, linestyle="--", label="EMA12")
    ax1.plot(dates_all, df_range["EMA26"], color="#2ca02c", linewidth=1, linestyle="--", label="EMA26")
    if len(golden) > 0:
        ax1.scatter(pd.to_datetime(golden["trade_date"]), golden["close"], color="red", s=60, marker="^", zorder=5, label="金叉")
    if len(death) > 0:
        ax1.scatter(pd.to_datetime(death["trade_date"]), death["close"], color="green", s=60, marker="v", zorder=5, label="死叉")
    ax1.set_ylabel("价格")
    ax1.set_title(f"{ts_code} MACD 金叉/死叉分析")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates_all, df_range["DIF"], color="#1f77b4", linewidth=1.5, label="DIF")
    ax2.plot(dates_all, df_range["DEA"], color="#ff7f0e", linewidth=1.5, linestyle="--", label="DEA")
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    if len(golden) > 0:
        ax2.scatter(pd.to_datetime(golden["trade_date"]), golden["DIF"], color="red", s=60, marker="^", zorder=5, label="金叉")
    if len(death) > 0:
        ax2.scatter(pd.to_datetime(death["trade_date"]), death["DIF"], color="green", s=60, marker="v", zorder=5, label="死叉")
    ax2.set_ylabel("DIF / DEA")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    colors = ["#d62728" if v >= 0 else "#2ca02c" for v in df_range["MACD"]]
    ax3.bar(dates_all, df_range["MACD"], color=colors, width=1.0)
    ax3.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax3.set_ylabel("MACD 柱")
    ax3.set_xlabel("日期")
    ax3.grid(True, alpha=0.3)

    tick_idx = _chart_tick_indexes(len(dates_all))
    ax3.set_xticks([dates_all.iloc[i] for i in tick_idx])
    ax3.set_xticklabels([dates_all.iloc[i].strftime("%Y-%m-%d") for i in tick_idx], rotation=45)

    fig.tight_layout()

    os.makedirs(_IMAGE_DIR, exist_ok=True)
    filename = f"macd_{datetime.now().strftime('%H%M%S%f')}.png"
    fig.savefig(os.path.join(_IMAGE_DIR, filename), dpi=200)
    plt.close(fig)
    return f"{result}![MACD 分析图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# ---- 工具5: 数据库数据更新 ----

@tool
def update_stock_data() -> str:
    """从 Tushare 获取增量数据并更新数据库中的股票每日价格。无需参数。"""
    token = os.environ.get("tushare_token")
    if not token:
        return "错误：未设置环境变量 tushare_token"
    ts.set_token(token)
    pro = ts.pro_api()

    engine = _get_engine()
    try:
        stocks_df = pd.read_sql(
            text("SELECT DISTINCT stock_name, ts_code FROM stock_daily_prices ORDER BY stock_name"),
            engine,
        )
    except Exception as e:
        return f"查询数据库股票列表失败: {str(e)}"
    finally:
        engine.dispose()

    if stocks_df.empty:
        return "数据库中没有任何股票数据"

    results = []
    total_new = 0

    for stock_name, ts_code in zip(stocks_df["stock_name"], stocks_df["ts_code"]):
        engine = _get_engine()
        try:
            max_date_df = pd.read_sql(
                text(f"SELECT MAX(trade_date) as max_date FROM stock_daily_prices WHERE ts_code='{ts_code}'"),
                engine, parse_dates=["max_date"],
            )
            max_date = max_date_df["max_date"].iloc[0]
            if max_date is None:
                continue

            start_str = (max_date + timedelta(days=1)).strftime("%Y%m%d")
            today_str = datetime.now().strftime("%Y%m%d")
            if start_str >= today_str:
                results.append(f"{stock_name}({ts_code}): 已是最新")
                continue

            time.sleep(0.3)
            try:
                df_new = pro.daily(ts_code=ts_code, start_date=start_str, end_date=today_str)
            except Exception as e:
                results.append(f"{stock_name}({ts_code}): Tushare 获取失败 - {str(e)}")
                continue

            if df_new.empty:
                results.append(f"{stock_name}({ts_code}): 无新数据")
                continue

            df_new["stock_name"] = stock_name
            df_new["trade_date"] = pd.to_datetime(df_new["trade_date"])
            df_new = df_new[["ts_code", "trade_date", "stock_name", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]]

            existing = pd.read_sql(
                text(f"SELECT trade_date FROM stock_daily_prices WHERE ts_code='{ts_code}'"),
                engine, parse_dates=["trade_date"],
            )
            existing_dates = set(existing["trade_date"].dt.date.astype(str))
            df_new = df_new[~df_new["trade_date"].dt.date.astype(str).isin(existing_dates)]

            if df_new.empty:
                results.append(f"{stock_name}({ts_code}): 无新数据（均已存在）")
                continue

            df_new.to_sql("stock_daily_prices", engine, if_exists="append", index=False, method="multi")
            total_new += len(df_new)
            results.append(f"{stock_name}({ts_code}): 新增 {len(df_new)} 条（{df_new['trade_date'].min():%Y-%m-%d} ~ {df_new['trade_date'].max():%Y-%m-%d}）")
        except Exception as e:
            results.append(f"{stock_name}({ts_code}): 失败 - {str(e)}")
        finally:
            engine.dispose()

    return f"## 数据更新完成\n\n共检查 {len(stocks_df)} 只股票，新增 {total_new} 条\n\n" + "\n".join(f"- {r}" for r in results)


# ---- 工具6: FAQ 知识库检索 ----

_search_faq_retriever = None

@tool
def search_faq(query: str) -> str:
    """在本地知识库中检索与当前问题相关的内容，知识库包含数据库表结构、股票列表、SQL示例和分析方法论。
    query: 搜索关键词或问题描述。
    """
    if _search_faq_retriever is None:
        return "FAQ 知识库不可用（未找到 faq.txt 文件）。"
    results = _search_faq_retriever.invoke(query)
    if not results:
        return "在 FAQ 知识库中未找到相关内容（所有切片 L2 距离均 >= 1.0）。"
    return "\n\n---\n\n".join(
        f"【{doc.metadata.get('source', 'faq.txt')}】（向量空间 L2 距离: {dist:.4f}）\n{doc.page_content}"
        for doc, dist in results
    )


# ---- 工具7: Tavily 联网搜索（在 init_agent 中惰性创建） ----


# ============================================================
# 7. Agent 组装
#
# create_react_agent 接收 model(LLM)、tools(工具列表)、prompt(system_prompt)，内部自动管理 ReAct 循环。
# ============================================================

def init_agent():
    """初始化并返回 LangChain Agent 实例。"""
    global _search_faq_retriever

    llm = ChatDeepSeekThink(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("deepseek_api"),
        timeout=300,
        extra_body={"thinking": {"type": "enabled"}},
    )

    _search_faq_retriever = _build_faq_retriever()

    tools = [exc_sql, predict_arima, detect_boll, detect_macd, update_stock_data, search_faq]

    if os.environ.get("tavily_api"):
        tools.append(TavilySearch(max_results=8, tavily_api_key=os.environ["tavily_api"]))

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    return agent


# ============================================================
# 8. Gradio 启动
# ============================================================

_SUGGESTIONS = [
    "查询2025年全年贵州茅台的收盘价走势",
    "统计2025年4月广发证券的日均成交量",
    "对比2025年全年中芯国际和贵州茅台的涨跌幅",
    "查询中芯国际的新闻动态",
    "使用ARIMA预测未来五天贵州茅台的收盘价",
    "检测贵州茅台过去一年布林带",
    "分析中芯国际近半年 MACD 金叉死叉信号",
    "更新一下数据库中的数据",
    "画一下中芯国际一年内的布林带，再展示一下广发证券的半年走势",
]

def _chat(message: str, history: list):
    """Gradio 回调：转换历史 → 向量相似度分析 → 调用 Agent → 补图 → HTML 化 → 返回。"""
    # 将 Gradio history 转为 LangChain 消息格式
    messages = []
    if history:
        for h in history:
            role = h.get("role", "")
            content = h.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    similarity_info = ""
    if _search_faq_retriever is not None:
        all_results = _search_faq_retriever.all_distances(message)
        if all_results:
            lines = [
                "> **向量知识库相似度分析**",
                f"> 查询: {message}",
                "> ",
                "> | 知识切片 | L2 距离 | 是否匹配 (< 1.0) |",
                "> |---|---|---|",
            ]
            for doc, dist in all_results:
                matched = "是" if dist < 1.0 else "否"
                source = doc.metadata.get("source", "未知")
                lines.append(f"> | {source} | {dist:.4f} | {matched} |")
            similarity_info = "\n".join(lines) + "\n\n---\n\n"

    # 调用 Agent：invoke 内部自动执行 ReAct 循环
    result = _agent.invoke({"messages": messages})

    # 从所有 ToolMessage 中提取工具生成的图片链接
    tool_images = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            tool_images.extend(re.findall(r'!\[.*?\]\(http://localhost:\d+/[^)]+\)', msg.content))

    # 取最后一条有内容的 assistant 消息作为最终回复
    final_content = None
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            final_content = msg.content
            break

    if final_content is None:
        return "（未获取到回复）"

    # 将 LLM 遗漏的图片链接补回末尾（DeepSeek 偶尔会省略）
    missing = [url for url in tool_images if url not in final_content]
    if missing:
        final_content += "\n\n" + "\n\n".join(missing)

    # 将 markdown 图片 ![](url) 转为带宽度控制的 HTML <img>，在聊天框中展示得更大
    final_content = re.sub(
        r'!\[(.*?)\]\((http://localhost:\d+/[^)]+)\)',
        r'<img src="\2" alt="\1" style="max-width:95%; width:100%; display:block; margin:10px auto;">',
        final_content,
    )

    if similarity_info:
        final_content = similarity_info + final_content

    return final_content


def app_gui():
    """启动顺序：图片服务器 → 初始化 Agent → 启动 Gradio 聊天界面。
    三个步骤有依赖关系，不可调换。"""
    global _agent
    _start_image_server()
    _agent = init_agent()

    demo = gr.ChatInterface(
        fn=_chat,
        title="股票查询助手（LangChain 版）",
        description="股票历史价格查询与分析",
        examples=_SUGGESTIONS,
        type="messages",
    )
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    app_gui()
