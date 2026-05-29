import os
import httpx
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
BASE_URL = os.getenv("BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
USE_GEMINI_SDK = os.getenv("USE_GEMINI_SDK", "false").lower() in ("true", "1", "yes")


def _detect_provider(base_url: str) -> str:
    """根据 BASE_URL 自动识别 AI 提供商。

    返回值：
        "gemini-official"  - Google 官方 Gemini API
        "openai"          - OpenAI API (api.openai.com)
        "custom"           - 第三方兼容接口（自定义域名）
    """
    if not base_url:
        return "gemini-official"
    lower = base_url.lower()
    if "generativelanguage.googleapis.com" in lower:
        return "gemini-official"
    if "api.openai.com" in lower:
        return "openai"
    return "custom"


class AIAnalyzer:
    """
    AI 分析类，支持多模型智能路由：
      - BASE_URL 为空        → Google 官方 Gemini SDK
      - generativelanguage  → Gemini REST API（httpx 直调）
      - api.openai.com      → OpenAI REST API（httpx 直调）
      - 其他自定义域名       → httpx 直调（自行拼接 /chat/completions）
    """

    def __init__(self):
        self._client = None
        self._mode = None

        if not API_KEY:
            print("警告: 未设置 API_KEY 环境变量，AI 分析功能将无法使用")
            return

        provider = _detect_provider(BASE_URL)

        # 模式一：Google 官方 Gemini SDK（绕过 BASE_URL）
        if USE_GEMINI_SDK:
            from google import genai
            from google.genai import types
            self._genai_types = types
            self._client = genai.Client(api_key=API_KEY)
            self._mode = "gemini-sdk"
            print(f"[AI] 模式: Gemini SDK，模型: {MODEL_NAME}")
            return

        # 模式二：Google Gemini REST API
        if provider == "gemini-official":
            self._mode = "gemini-rest"
            print(f"[AI] 模式: Gemini REST API，BASE_URL: {BASE_URL or '(默认)'}，模型: {MODEL_NAME}")
            return

        # 模式三：OpenAI REST API（标准路径 /v1/chat/completions）
        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            self._mode = "openai-sdk"
            print(f"[AI] 模式: OpenAI SDK，base_url: {BASE_URL}，模型: {MODEL_NAME}")
            return

        # 模式四：第三方兼容接口（自定义域名，httpx 直调，自行拼接 /chat/completions）
        self._mode = "custom-httpx"
        print(f"[AI] 模式: 自定义代理（httpx），base_url: {BASE_URL}，模型: {MODEL_NAME}")

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
        """
        if not API_KEY:
            return "错误: 未设置 API_KEY 环境变量，无法使用 AI 分析功能。请在 .env 文件中添加 API_KEY。"

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
            mode = self._mode or "unknown"
            if mode == "gemini-sdk":
                analysis_result = self._call_gemini_sdk(prompt, img_bytes)
            elif mode == "gemini-rest":
                analysis_result = self._call_gemini_rest(prompt, img_bytes)
            elif mode == "openai-sdk":
                analysis_result = self._call_openai_sdk(prompt, img_base64)
            elif mode == "custom-httpx":
                analysis_result = self._call_custom_httpx(prompt, img_base64)
            else:
                return f"错误: 未知的 AI 调用模式: {mode}"

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

    # ── 代理支持 ──────────────────────────────────────────────────────────────

    def _build_client(self) -> httpx.Client:
        """构造 httpx 客户端，自动读取 HTTPS_PROXY 环境变量。"""
        proxies = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        timeout = httpx.Timeout(120.0, connect=30.0)
        if proxies:
            return httpx.Client(proxies=proxies, timeout=timeout)
        return httpx.Client(timeout=timeout)

    # ── Gemini 官方 SDK ───────────────────────────────────────────────────────

    def _call_gemini_sdk(self, prompt: str, img_bytes: bytes) -> str:
        """通过 Google 官方 Gemini SDK 调用。"""
        response = self._client.models.generate_content(
            model=MODEL_NAME,
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

    # ── Gemini REST API ───────────────────────────────────────────────────────

    def _call_gemini_rest(self, prompt: str, img_bytes: bytes) -> str:
        """通过 Gemini REST API 调用（httpx 直调）。"""
        # Gemini REST API 固定路径格式：/v1beta/models/{model}:generateContent
        # API key 通过 x-goog-api-key header 传递（不放在 URL 里）
        base = BASE_URL.rstrip('/') if BASE_URL else "https://generativelanguage.googleapis.com"
        url = f"{base}/v1beta/models/{MODEL_NAME}:generateContent"

        headers = {
            "x-goog-api-key": API_KEY,
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_bytes).decode("utf-8")}}
                ]
            }],
            "systemInstruction": {
                "parts": [{"text": "你是一位专业的股票分析师，请基于以下数据分析股票的K线图和基本面情况，并预测上涨的概率。"}]
            },
            "generationConfig": {
                "temperature": 0,
                "topP": 0.95,
                "candidateCount": 1,
            }
        }

        with self._build_client() as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

        # 解析响应文本
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return str(result)

    # ── OpenAI SDK ────────────────────────────────────────────────────────────

    def _call_openai_sdk(self, prompt: str, img_base64: str) -> str:
        """通过 OpenAI SDK 调用（api.openai.com）。"""
        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的股票分析师，请基于以下数据分析股票的K线图和基本面情况，并预测上涨的概率。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            temperature=0,
            top_p=0.95,
        )
        return response.choices[0].message.content

    # ── 第三方兼容接口（自定义域名）──────────────────────────────────────────

    def _call_custom_httpx(self, prompt: str, img_base64: str) -> str:
        """通过 httpx 直调第三方兼容接口，自行拼接 /chat/completions。"""
        if not BASE_URL:
            return "错误: BASE_URL 未配置，无法使用自定义代理模式。"

        url = f"{BASE_URL.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的股票分析师，请基于以下数据分析股票的K线图和基本面情况，并预测上涨的概率。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            "temperature": 0,
            "top_p": 0.95,
        }

        with self._build_client() as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

        return result["choices"][0]["message"]["content"]

    # ── 数据准备 ──────────────────────────────────────────────────────────────

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
