import os
from PIL import Image
import pandas as pd
import json
from datetime import datetime
import io
import base64
import threading
import logging
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(message)s')

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "")          # 如留空则用 Gemini SDK
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
USE_GEMINI_SDK = os.getenv("USE_GEMINI_SDK", "false").lower() in ("true", "1", "yes")


class AIAnalyzer:
    """
    AI分析类，支持两种调用模式：
      1. OpenAI 兼容接口（通过 BASE_URL 配置）
      2. Google 官方 Gemini SDK（直接调用 Gemini API）
    """

    def __init__(self):
        if not API_KEY:
            self._client = None
            self._mode = None
            print("警告: 未设置 API_KEY 环境变量，AI 分析功能将无法使用")
            return

        # 优先使用 OpenAI 兼容模式（BASE_URL 非空时）
        if USE_GEMINI_SDK:
            from google import genai
            from google.genai import types
            self._genai_types = types
            self._client = genai.Client(api_key=API_KEY)
            self._mode = "gemini-sdk"
            print(f"[AI] 使用 Gemini SDK 模式，模型: {MODEL_NAME}")
        elif BASE_URL:
            from openai import OpenAI
            self._client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            self._mode = "openai-compatible"
            print(f"[AI] 使用 OpenAI 兼容模式，base_url: {BASE_URL}，模型: {MODEL_NAME}")
        else:
            # 回退到 Gemini SDK
            from google import genai
            from google.genai import types
            self._genai_types = types
            self._client = genai.Client(api_key=API_KEY)
            self._mode = "gemini-sdk"
            print(f"[AI] 未配置 BASE_URL，自动切换到 Gemini SDK 模式，模型: {MODEL_NAME}")

    def _get_stock_name(self, stock_code: str) -> str:
        """获取股票名称，Baostock 优先，AKShare 备选。"""
        code = stock_code.lstrip('sz').lstrip('sh').lstrip('SZ').lstrip('SH').rstrip(
            '.sz').rstrip('.sh').rstrip('.SZ').rstrip('.SH').zfill(6)
        bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"

        try:
            import baostock as bs
            bs.login()
            rs = bs.query_stock_basic(code=bs_code)
            data = rs.get_data()
            bs.logout()
            if not data.empty and 'code_name' in data.columns:
                name = data['code_name'].values[0]
                if name:
                    return name
        except Exception:
            pass

        try:
            import akshare as ak
            stock_info = ak.stock_individual_info_em(symbol=code)
            if not stock_info.empty:
                name = stock_info.loc[
                    stock_info['item'] == '股票简称', 'value'
                ].values
                if len(name) > 0:
                    return name[0]
        except Exception:
            pass

        return stock_code

    def analyze(self, stock_data, indicators, financial_data, news_data, stock_code, save_path):
        """
        使用 AI 分析股票数据并预测未来走势。
        自动根据 .env 配置选择 OpenAI 兼容接口或 Gemini SDK。
        """
        if not API_KEY:
            return "错误: 未设置 API_KEY 环境变量，无法使用 AI 分析功能。请在 .env 文件中添加 API_KEY。"

        if self._client is None:
            return f"错误: AI 客户端初始化失败（模式: {self._mode}），请检查 .env 配置。"

        try:
            logging.debug(f"运行 analyze 方法的线程: {threading.current_thread().name}")

            stock_name = self._get_stock_name(stock_code)
            analysis_data = self._prepare_analysis_data(stock_data, indicators, financial_data, news_data,
                                                       stock_code, stock_name)
            prompt = self._build_prompt(analysis_data, stock_code, stock_name)

            # 读取并编码图片
            image_path = os.path.join(save_path, f"charts/{stock_code}_technical_analysis.png")
            img = Image.open(image_path)
            try:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            finally:
                img.close()

            # 根据模式调用对应的 API
            if self._mode == "gemini-sdk":
                analysis_result = self._call_gemini_sdk(prompt, img_bytes, MODEL_NAME)
            else:
                analysis_result = self._call_openai_compatible(prompt, img_base64, MODEL_NAME)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            disclaimer = (
                "\n\n免责声明：本分析报告由 AI 自动生成，仅供参考，不构成任何投资建议。"
                "投资有风险，入市需谨慎。"
            )
            return (
                f"# {stock_name}({stock_code}) AI 预测概率\n\n"
                f"生成时间: {current_time}\n\n"
                f"{analysis_result}\n{disclaimer}"
            )

        except Exception as e:
            return f"AI 分析过程中出错: {str(e)}"

    def _call_gemini_sdk(self, prompt: str, img_bytes: bytes, model: str) -> str:
        """通过 Google 官方 Gemini SDK 调用。"""
        response = self._client.models.generate_content(
            model=model,
            contents=[
                self._genai_types.Content(
                    parts=[
                        self._genai_types.Part(text=prompt),
                        self._genai_types.Part(
                            inline_data=self._genai_types.Blob(
                                mime_type="image/png",
                                data=img_bytes
                            )
                        )
                    ]
                )
            ],
            config=self._genai_types.GenerateContentConfig(
                system_instruction="你是一位专业的股票分析师，请基于以下数据分析股票的K线图和基本面情况，并预测上涨的概率。",
                temperature=0,
                top_p=0.95,
                candidate_count=1,
            )
        )
        return response.text

    def _call_openai_compatible(self, prompt: str, img_base64: str, model: str) -> str:
        """通过 OpenAI 兼容接口调用（BASE_URL + OpenAI SDK）。"""
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的股票分析师，请基于以下数据分析股票的K线图和基本面情况，并预测上涨的概率。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            temperature=0,
            top_p=0.95,
        )
        return response.choices[0].message.content

    def _prepare_analysis_data(self, stock_data, indicators, financial_data, news_data,
                               stock_code, stock_name):
        """准备用于分析的数据。"""
        analysis_data = {}
        analysis_data['股票代码'] = stock_code
        analysis_data['股票名称'] = stock_name

        if not stock_data.empty:
            latest = stock_data.iloc[-1]
            earliest = stock_data.iloc[0]

            analysis_data['当前价格'] = float(latest['close'])
            analysis_data['开盘价'] = float(latest['open'])
            analysis_data['最高价'] = float(latest['high'])
            analysis_data['最低价'] = float(latest['low'])
            analysis_data['成交量'] = float(latest['volume'])
            analysis_data['日期'] = latest['date'].strftime('%Y-%m-%d')

            price_change = (latest['close'] - earliest['close']) / earliest['close'] * 100
            analysis_data['区间涨跌幅'] = round(price_change, 2)

            recent_days = min(30, len(stock_data))
            analysis_data['最近价格趋势'] = stock_data['close'].tail(recent_days).tolist()
            analysis_data['最近成交量趋势'] = stock_data['volume'].tail(recent_days).tolist()
            analysis_data['最近日期'] = [d.strftime('%Y-%m-%d') for d in stock_data['date'].tail(recent_days)]

        if indicators:
            latest_idx = -1
            for key in ('MA5', 'MA10', 'MA20', 'MA30', 'MACD', 'Signal', 'Histogram',
                        'K', 'D', 'J', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower'):
                if key in indicators and not indicators[key].empty:
                    val = indicators[key].iloc[latest_idx]
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        analysis_data[key] = round(float(val), 4)

        if financial_data:
            analysis_data['财务数据'] = financial_data

        if news_data:
            analysis_data['新闻'] = news_data[:5]

        return analysis_data

    def _build_prompt(self, analysis_data: dict, stock_code: str, stock_name: str) -> str:
        """构建发送给 AI 的提示词。"""
        prompt_parts = [
            f"请分析股票 {stock_name}（代码: {stock_code}）的技术面和基本面情况：\n",
            json.dumps(analysis_data, ensure_ascii=False, indent=2),
            "\n请给出：1. 技术面综合判断；2. 基本面简评（如有）；3. 未来趋势预测；4. 上涨概率（0%-100%）。"
        ]
        return "\n".join(prompt_parts)
