"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Qwen-Agent 标准教材 —— 股票查询助手                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  学习地图：每一个模块对应 Agent 开发的一个核心概念。                          ║
║                                                                              ║
║  [模块 1] 导入            → 知道你需要哪些库                                  ║
║  [模块 2] 全局配置        → 配置 matplotlib、图片目录                         ║
║  [模块 3] 辅助函数        → 数据库引擎、图片服务器、图表绘制                   ║
║  [模块 4] system_prompt   → ★ 这是 Agent 的"大脑说明书"                       ║
║  [模块 5] 工具定义        → ★ 注册自定义工具（@register_tool）                ║
║  [模块 6] Agent 初始化     → ★ 组装：LLM + system_prompt + function_list      ║
║  [模块 7] 启动            → WebUI 可视化界面                                  ║
║                                                                              ║
║  运行方式：                                                                   ║
║    设置环境变量后直接执行本文件：                                              ║
║      deepseek_api   (DeepSeek API 密钥)                                       ║
║      MySQL_key      (本地 MySQL root 密码)                                    ║
║      tushare_token  (Tushare 数据平台 token)                                  ║
║      tavily_api     (Tavily 搜索 API 密钥)                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# [模块 1] 导入
# ============================================================
# 标准库
import http.server
import json
import os
import socket
import threading
from datetime import datetime, timedelta

# 数据处理与可视化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
from sqlalchemy import create_engine, text
from statsmodels.tsa.arima.model import ARIMA

# qwen_agent 框架核心
from qwen_agent.agents import Assistant          # Agent 主体
from qwen_agent.gui import WebUI                  # 聊天界面
from qwen_agent.tools.base import BaseTool, register_tool  # 工具基类与注册器


# ============================================================
# [模块 2] 全局配置
# ============================================================

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "image_show")
_IMAGE_SERVER_PORT = None


# ============================================================
# [模块 3] 辅助函数
# ============================================================
# 这些是工具函数的基础设施，不属于 Agent 框架，但工具会调用它们。

def _get_engine():
    """创建 MySQL 数据库连接引擎（连接池复用）。"""
    return create_engine(
        f"mysql+pymysql://root:{os.environ['MySQL_key']}@localhost:3306/chatbi_data?charset=utf8mb4",
        connect_args={"connect_timeout": 10},
        pool_size=5,
        max_overflow=10,
    )


def _start_image_server():
    """在随机端口启动图片 HTTP 服务器，让 WebUI 能加载本地生成的图表。"""
    global _IMAGE_SERVER_PORT
    os.makedirs(_IMAGE_DIR, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        _IMAGE_SERVER_PORT = sock.getsockname()[1]

    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=_IMAGE_DIR, **kw)
    server = http.server.HTTPServer(("localhost", _IMAGE_SERVER_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def _chart_tick_indexes(total: int, max_ticks: int = 10) -> np.ndarray:
    """计算图表 X 轴刻度位置，避免标签过密重叠。"""
    if total <= max_ticks:
        return np.arange(total)
    step = max(1, total // max_ticks)
    idx = np.arange(0, total, step)
    return np.append(idx, total - 1) if idx[-1] != total - 1 else idx


def generate_chart_png(df: pd.DataFrame, save_path: str):
    """将 DataFrame 绘制为 PNG 走势图并保存。

    判断规则：
      - 有日期列 → 折线图（X 轴为日期）
      - 无日期列 → 柱状图（≤20条）或折线图（>20条）
    """
    if df.empty:
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not num_cols:
        return

    # 检测第一列是否为日期
    date_col = None
    if non_num_cols:
        try:
            pd.to_datetime(df[non_num_cols[0]], errors="raise")
            date_col = non_num_cols[0]
        except Exception:
            pass

    n = len(df)
    fig, ax = plt.subplots(figsize=(12, 6))

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
                labels = df[label_col].astype(str).iloc[tick_idx].tolist()
                ax.set_xticklabels([s.replace("%", "%%") for s in labels], rotation=45)
        else:
            ax.bar(range(n), df[num_cols[0]], label=num_cols[0])
            if label_col:
                ax.set_xticks(range(n))
                labels = df[label_col].astype(str).tolist()
                ax.set_xticklabels([s.replace("%", "%%") for s in labels], rotation=45)
        ax.set_ylabel(num_cols[0])
        ax.set_title("股票数据")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# [模块 4] system_prompt —— Agent 的"行为说明书"
# ============================================================
# ★ 这是 Agent 最重要的配置。LLM 通过它理解：
#    1. 自己是谁（角色）
#    2. 知道什么（数据库结构、股票列表）
#    3. 什么场景用什么工具
#    4. 回答时要注意什么（如图片不能省略）

system_prompt = """我是股票查询助手，可以查询股票历史价格数据。以下是数据库表结构：

-- 股票每日价格表
CREATE TABLE stock_daily_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL,
    stock_name VARCHAR(50) NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE,
    vol DOUBLE, amount DOUBLE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

可查询的股票：贵州茅台(600519.SH)、五粮液(000858.SZ)、广发证券(000776.SZ)、中芯国际(688981.SH)
数据范围：2020-01-02 至今

常见查询示例：
1. SELECT trade_date, close FROM stock_daily_prices WHERE stock_name='贵州茅台' AND trade_date BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY trade_date;
2. SELECT MAX(high), MIN(low), AVG(close) FROM stock_daily_prices WHERE stock_name='贵州茅台' AND trade_date BETWEEN '2024-01-01' AND '2024-12-31';
3. SELECT trade_date, vol, pct_chg FROM stock_daily_prices WHERE stock_name='中芯国际' AND trade_date BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY trade_date;
4. SELECT trade_date, stock_name, vol FROM stock_daily_prices WHERE trade_date BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY vol DESC LIMIT 10;

工具使用场景：
- ARIMA 预测：用户说"预测未来N天股价"时使用 arima_stock 工具，传入 ts_code 和 n。
- BOLL 布林带：用户说"检测超买超卖、布林带"时使用 boll_detection 工具，传入 ts_code。
- 联网搜索：涉及实时信息或数据库之外的知识时，使用 tavily_search 工具。
- 数据更新：用户说"更新数据"时，使用 update_stock_data 工具，无需参数直接调用。

【重要输出规则】
exc_sql、boll_detection、arima_stock 这三个工具返回的结果中可能包含图片（格式为 ![](url)），你必须原样保留所有图片 markdown 链接，严禁省略、删除或以文字替代图片。
当用户请求涉及多个分析任务（如同时要求布林带检测和走势图）时，你必须并列输出每个任务的完整结果（包括每个任务各自的图片），不允许只保留一个图片或合并输出。
如果不确定是否应包含图片，优先保留图片。"""


# ============================================================
# [模块 5] 工具定义 —— Agent 的"双手"
# ============================================================
# ★ 每个工具遵循统一结构：
#    @register_tool("工具名")     ← 注册到全局工具表，function_list 中用这个名字引用
#    class XxxTool(BaseTool):
#        description = "..."      ← LLM 据此判断何时调用此工具
#        parameters = [...]       ← 定义 LLM 需要提供哪些参数（name/type/description/required）
#        def call(params, **kw)   ← 执行入口，params 是 LLM 生成的 JSON 字符串
#
# 工具返回的字符串会直接喂给 LLM 作为"工具执行结果"，
# LLM 再据此组织最终回复。支持 Markdown，可以包含图片、表格。

# --- 工具 ①：SQL 查询与可视化 ---

@register_tool("exc_sql")
class ExcSQLTool(BaseTool):
    """执行 SQL 查询，返回表格数据 + 描述统计 + 走势图。"""
    description = "执行 SQL 查询，并以表格 + 图表形式返回查询结果"
    parameters = [
        {
            "name": "sql_input",
            "type": "string",
            "description": "要执行的 SQL 查询语句",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        sql = args["sql_input"]

        engine = _get_engine()
        try:
            df = pd.read_sql(text(sql), engine)
        except Exception as e:
            return f"SQL 执行失败: {str(e)}"
        finally:
            engine.dispose()

        # 构建 Markdown 表格（数据 ≤10 行全量展示，否则首尾各 5 行）
        if df.shape[0] <= 10:
            md = df.to_markdown(index=False)
        else:
            md = (
                f"{df.head(5).to_markdown(index=False)}\n\n"
                f"...（中间省略 {df.shape[0] - 10} 行）...\n\n"
                f"{df.tail(5).to_markdown(index=False)}"
            )
        md += f"\n\n**描述统计:**\n{df.describe().to_markdown()}"

        # 单行数据无需画图
        if df.shape[0] == 1:
            return md

        os.makedirs(_IMAGE_DIR, exist_ok=True)
        filename = f"chart_{datetime.now().strftime('%H%M%S%f')}.png"
        save_path = os.path.join(_IMAGE_DIR, filename)
        generate_chart_png(df, save_path)
        return f"{md}\n\n![图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# --- 工具 ②：ARIMA 股价预测 ---

@register_tool("arima_stock")
class ArimaStockTool(BaseTool):
    """基于 ARIMA(5,1,5) 统计模型预测未来 N 个交易日收盘价。"""
    description = "使用 ARIMA(5,1,5) 模型预测股票未来 N 天的收盘价，输入 ts_code 和预测天数 n"
    parameters = [
        {"name": "ts_code", "type": "string", "description": "股票代码，例如 600519.SH", "required": True},
        {"name": "n", "type": "integer", "description": "预测未来多少天的收盘价", "required": True},
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        ts_code = args["ts_code"]
        n = args["n"]

        engine = _get_engine()
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=366)).strftime("%Y-%m-%d")
            df = pd.read_sql(
                text(f"""SELECT trade_date, close FROM stock_daily_prices
                        WHERE ts_code='{ts_code}' AND trade_date BETWEEN '{start}' AND '{end}'
                        ORDER BY trade_date"""),
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

        # 预测日期（跳过周末）
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

        # 绘制历史 + 预测对比图
        hist_dates = pd.to_datetime(df["trade_date"])
        pred_dt = pd.to_datetime(pred_dates)
        tail = min(len(hist_dates), 120)

        fig, ax = plt.subplots(figsize=(12, 6))
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
        plt.tight_layout()

        os.makedirs(_IMAGE_DIR, exist_ok=True)
        filename = f"arima_{datetime.now().strftime('%H%M%S%f')}.png"
        plt.savefig(os.path.join(_IMAGE_DIR, filename), dpi=150)
        plt.close()
        return f"{summary}\n\n![ARIMA 预测图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# --- 工具 ③：布林带异常检测 ---

@register_tool("boll_detection")
class BollDetectionTool(BaseTool):
    """布林带（Bollinger Bands）技术指标检测：
       - 中轨 = 20 日移动平均线
       - 上轨 = 中轨 + 2σ（超买警戒线）
       - 下轨 = 中轨 - 2σ（超卖警戒线）"""
    description = "使用布林带（BOLL，20 日周期 + 2σ）检测股票的超买和超卖异常点"
    parameters = [
        {"name": "ts_code", "type": "string", "description": "股票代码，例如 600519.SH", "required": True},
        {"name": "start_date", "type": "string", "description": "开始日期，格式 YYYY-MM-DD，可选，默认 1 年前", "required": False},
        {"name": "end_date", "type": "string", "description": "结束日期，格式 YYYY-MM-DD，可选，默认今天", "required": False},
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        ts_code = args["ts_code"]
        start_date = args.get("start_date") or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = args.get("end_date") or datetime.now().strftime("%Y-%m-%d")

        engine = _get_engine()
        try:
            # 多往前取 60 天，保证 20 日滚动窗口有足够数据
            extended_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime("%Y-%m-%d")
            df = pd.read_sql(
                text(f"""SELECT trade_date, close FROM stock_daily_prices
                        WHERE ts_code='{ts_code}' AND trade_date BETWEEN '{extended_start}' AND '{end_date}'
                        ORDER BY trade_date"""),
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

        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        df_range = df[(df["trade_date"] >= sd) & (df["trade_date"] <= ed)].dropna(subset=["MA20", "upper", "lower"]).copy()
        if df_range.empty:
            return f"在 {start_date} ~ {end_date} 内无足够数据计算布林带"

        df_range["overbought"] = df_range["close"] > df_range["upper"]
        df_range["oversold"] = df_range["close"] < df_range["lower"]
        ob = df_range[df_range["overbought"]]
        os_rows = df_range[df_range["oversold"]]

        result = f"基于布林带（20 日 + 2σ）对 {ts_code} 在 {start_date} ~ {end_date} 的检测结果：\n\n"
        result += f"**检测统计：**\n- 总交易日：{len(df_range)}\n- 超买天数：{len(ob)}\n- 超卖天数：{len(os_rows)}\n\n"

        if len(ob) > 0:
            tbl = ob[["trade_date", "close", "upper"]].copy()
            tbl.columns = ["交易日期", "收盘价", "上轨"]
            result += f"**超买信号（收盘价 > 上轨）：**\n{tbl.to_markdown(index=False)}\n\n"
        else:
            result += "**超买信号：** 无\n\n"

        if len(os_rows) > 0:
            tbl = os_rows[["trade_date", "close", "lower"]].copy()
            tbl.columns = ["交易日期", "收盘价", "下轨"]
            result += f"**超卖信号（收盘价 < 下轨）：**\n{tbl.to_markdown(index=False)}\n\n"
        else:
            result += "**超卖信号：** 无\n\n"

        # 绘制布林带走势图
        dates_all = pd.to_datetime(df_range["trade_date"])
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(dates_all, df_range["upper"], df_range["lower"], alpha=0.1, color="gray")
        ax.plot(dates_all, df_range["close"], color="#1f77b4", linewidth=2.5, label="收盘价")
        ax.plot(dates_all, df_range["MA20"], color="#ff7f0e", linewidth=1.5, linestyle="--", label="中轨(MA20)")
        ax.plot(dates_all, df_range["upper"], color="#d62728", linewidth=1.5, linestyle="--", label="上轨")
        ax.plot(dates_all, df_range["lower"], color="#d62728", linewidth=1.5, linestyle="--", label="下轨")

        if len(ob) > 0:
            ax.scatter(pd.to_datetime(ob["trade_date"]), ob["close"], color="red", s=50, marker="v", zorder=5, label="超买点")
        if len(os_rows) > 0:
            ax.scatter(pd.to_datetime(os_rows["trade_date"]), os_rows["close"], color="green", s=50, marker="^", zorder=5, label="超卖点")

        tick_idx = _chart_tick_indexes(len(dates_all))
        ax.set_xticks([dates_all.iloc[i] for i in tick_idx])
        ax.set_xticklabels([dates_all.iloc[i].strftime("%Y-%m-%d") for i in tick_idx], rotation=45)
        ax.set_ylabel("价格")
        ax.set_title(f"{ts_code} 布林带超买/超卖检测")
        ax.legend(loc="best")
        plt.tight_layout()

        os.makedirs(_IMAGE_DIR, exist_ok=True)
        filename = f"boll_{datetime.now().strftime('%H%M%S%f')}.png"
        plt.savefig(os.path.join(_IMAGE_DIR, filename), dpi=150)
        plt.close()
        return f"{result}![布林带检测图表](http://localhost:{_IMAGE_SERVER_PORT}/{filename})"


# --- 工具 ④：数据库数据更新 ---

@register_tool("update_stock_data")
class UpdateStockDataTool(BaseTool):
    """通过 Tushare API 增量更新数据库中的股票每日价格。"""
    description = "从 Tushare 获取增量数据并更新数据库中的股票每日价格"
    parameters = []  # 无参数工具，LLM 直接调用

    def call(self, params: str, **kwargs) -> str:
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
                # ① 查询该股票的最新数据日期
                max_date_df = pd.read_sql(
                    text(f"SELECT MAX(trade_date) as max_date FROM stock_daily_prices WHERE ts_code='{ts_code}'"),
                    engine,
                    parse_dates=["max_date"],
                )
                max_date = max_date_df["max_date"].iloc[0]
                if max_date is None:
                    continue

                start_str = (max_date + timedelta(days=1)).strftime("%Y%m%d")
                today_str = datetime.now().strftime("%Y%m%d")
                if start_str >= today_str:
                    results.append(f"{stock_name}({ts_code}): 已是最新")
                    continue

                # ② 从 Tushare 拉取增量数据
                threading.Event().wait(0.3)
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

                # ③ 过滤已存在的日期（避免重复插入）
                existing = pd.read_sql(
                    text(f"SELECT trade_date FROM stock_daily_prices WHERE ts_code='{ts_code}'"),
                    engine,
                    parse_dates=["trade_date"],
                )
                existing_dates = set(existing["trade_date"].dt.date.astype(str))
                df_new = df_new[~df_new["trade_date"].dt.date.astype(str).isin(existing_dates)]

                if df_new.empty:
                    results.append(f"{stock_name}({ts_code}): 无新数据（均已存在）")
                    continue

                # ④ 写入数据库
                df_new.to_sql("stock_daily_prices", engine, if_exists="append", index=False, method="multi")
                total_new += len(df_new)
                results.append(
                    f"{stock_name}({ts_code}): 新增 {len(df_new)} 条（{df_new['trade_date'].min():%Y-%m-%d} ~ {df_new['trade_date'].max():%Y-%m-%d}）"
                )
            except Exception as e:
                results.append(f"{stock_name}({ts_code}): 失败 - {str(e)}")
            finally:
                engine.dispose()

        return f"## 数据更新完成\n\n共检查 {len(stocks_df)} 只股票，新增 {total_new} 条\n\n" + "\n".join(f"- {r}" for r in results)


# ============================================================
# [模块 6] Agent 初始化 —— ★ 核心：把大脑和双手组装起来
# ============================================================
# Assistant 是 qwen_agent 的核心类，它内部封装了 ReAct 循环：
#   思考(LLM) → 决策(调用哪个工具) → 执行(run tool) → 观察(拿到结果) → 再思考...
# 你只需配置三要素：
#   1. llm            — 用什么模型、怎么连接
#   2. system_message — 行为说明书（模块 4）
#   3. function_list  — 可用工具列表（模块 5 注册的工具名 + MCP 服务）

def init_agent_service():
    """初始化并返回 Agent 实例。"""
    llm_cfg = {
        "model": "deepseek-v4-flash",
        "model_server": "https://api.deepseek.com",
        "api_key": os.environ.get("deepseek_api"),
        "model_type": "oai",          # 以 OpenAI 兼容协议调用
        "timeout": 300,
        "generate_cfg": {
            "extra_body": {
                "thinking": {"type": "enabled"},  # 开启 DeepSeek 思考模式
            },
        },
    }

    bot = Assistant(
        llm=llm_cfg,
        name="股票查询助手",
        description="股票历史价格查询与分析",
        system_message=system_prompt,
        function_list=[
            "exc_sql",                # 自定义工具：字符串引用
            "arima_stock",
            "boll_detection",
            "update_stock_data",
            {                         # MCP 工具：字典配置
                "mcpServers": {
                    "tavily-mcp": {
                        "command": "npx",
                        "args": ["-y", "tavily-mcp@0.1.4"],
                        "autoApprove": [],
                        "env": {"TAVILY_API_KEY": os.getenv("tavily_api")},
                    },
                },
            },
        ],
        files=["faq.txt"],            # 附加知识文件
    )

    print("股票查询助手初始化成功！")
    return bot


# ============================================================
# [模块 7] 启动 —— WebUI 可视化聊天界面
# ============================================================
# WebUI 是 qwen_agent 内置的 Gradio 聊天界面。
# prompt.suggestions 是界面默认展示的建议问题。

def app_gui():
    _start_image_server()
    bot = init_agent_service()

    chatbot_config = {
        "prompt.suggestions": [
            "查询2025年全年贵州茅台的收盘价走势",
            "统计2025年4月广发证券的日均成交量",
            "对比2025年全年中芯国际和贵州茅台的涨跌幅",
            "查询中芯国际的新闻动态",
            "使用ARIMA预测未来五天贵州茅台的收盘价",
            "检测贵州茅台过去一年布林带",
            "更新一下数据库中的数据",
            "画一下中芯国际一年内的布林带，再展示一下广发证券的半年走势",
        ]
    }

    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == "__main__":
    app_gui()
