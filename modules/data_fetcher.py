import pandas as pd
from datetime import datetime, timedelta

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except Exception:
    AKSHARE_AVAILABLE = False

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except Exception:
    BAOSTOCK_AVAILABLE = False


class StockDataFetcher:
    """
    股票数据获取类，支持 AKShare（主） + Baostock（备）双数据源自动切换。
    """

    def __init__(self):
        self.today = datetime.now().strftime('%Y-%m-%d')
        self._bs_logged_in = False
        self._source = None

    # ------------------------------------------------------------------
    # 私有方法：Baostock 登录管理
    # ------------------------------------------------------------------
    def _bs_login(self):
        if BAOSTOCK_AVAILABLE and not self._bs_logged_in:
            result = bs.login()
            if result.error_code == '0':
                self._bs_logged_in = True

    def _bs_logout(self):
        if BAOSTOCK_AVAILABLE and self._bs_logged_in:
            try:
                bs.logout()
            except Exception:
                pass
            self._bs_logged_in = False

    # ------------------------------------------------------------------
    # 私有方法：统一列名
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
        """将不同数据源的列名统一为本项目内部格式。"""
        if source == 'akshare':
            rename_map = {
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude',
                '涨跌幅': 'pct_change', '涨跌额': 'change', '换手率': 'turnover'
            }
            df.rename(columns=rename_map, inplace=True)

        elif source == 'baostock':
            rename_map = {
                'date': 'date', 'open': 'open', 'close': 'close',
                'high': 'high', 'low': 'low', 'volume': 'volume',
                'pctChg': 'pct_change'
            }
            existing = {k: v for k, v in rename_map.items() if k in df.columns}
            df.rename(columns=existing, inplace=True)

            # Baostock 所有数值列都是带空格的字符串，统一转为 float
            numeric_cols = ('open', 'close', 'high', 'low', 'volume', 'pct_change')
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.strip(), errors='coerce'
                    )

            # Baostock 不提供以下字段，填充空值
            for col in ('amount', 'amplitude', 'change', 'turnover'):
                if col not in df.columns:
                    df[col] = None

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def fetch_stock_data(self, stock_code, period='1年'):
        """
        获取股票历史K线数据。自动尝试 AKShare → Baostock 备选。

        参数:
            stock_code (str): 股票代码，如 '000001'
            period (str): 获取周期，默认'1年'

        返回:
            pandas.DataFrame: 统一格式的K线数据
        """
        code = self._normalize_code(stock_code)
        start_date = self._period_to_start_date(period)

        # 优先 AKShare
        if AKSHARE_AVAILABLE:
            try:
                df = self._fetch_via_akshare(code, start_date)
                if df is not None and not df.empty:
                    self._source = 'akshare'
                    return df
            except Exception as e:
                print(f"  [数据源] AKShare 失败: {e}，切换到 Baostock...")

        # 备选 Baostock
        if BAOSTOCK_AVAILABLE:
            try:
                df = self._fetch_via_baostock(code, start_date)
                if df is not None and not df.empty:
                    self._source = 'baostock'
                    return df
            except Exception as e:
                print(f"  [数据源] Baostock 也失败: {e}")

        print(f"[数据源] 所有数据源均不可用，请检查网络环境。")
        return pd.DataFrame()

    def fetch_financial_data(self, stock_code):
        """获取财务数据，仅支持 AKShare，失败时返回空字典。"""
        if not AKSHARE_AVAILABLE:
            return {}

        financial_data = {}
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            if not stock_info.empty:
                financial_data['基本信息'] = stock_info.set_index('item').to_dict()['value']

            financial_abstract = ak.stock_financial_abstract(symbol=stock_code)
            if not financial_abstract.empty and '指标' in financial_abstract.columns:
                # 取第一列（指标名）作为 index，第二列作为 value
                value_col = financial_abstract.columns[1]
                financial_data['关键指标'] = (
                    financial_abstract[['指标', value_col]]
                    .dropna()
                    .set_index('指标')[value_col]
                    .to_dict()
                )

            return financial_data
        except Exception as e:
            print(f"获取财务数据时出错: {e}")
            return financial_data

    def fetch_news_data(self, stock_code, max_items=10):
        """获取新闻数据，仅支持 AKShare，失败时返回空列表。"""
        if not AKSHARE_AVAILABLE:
            return []

        news_list = []
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            if not stock_info.empty:
                stock_name = stock_info.loc[
                    stock_info['item'] == '股票简称', 'value'
                ].values[0]

                news_data = ak.stock_news_em(symbol=stock_code)
                if not news_data.empty:
                    for _, row in news_data.head(max_items).iterrows():
                        news_list.append({
                            'title': row['新闻标题'],
                            'date': row['发布时间'],
                            'content': row.get('新闻内容', ''),
                        })
            return news_list
        except Exception as e:
            print(f"获取新闻数据时出错: {e}")
            return news_list

    # ------------------------------------------------------------------
    # 私有方法：数据获取实现
    # ------------------------------------------------------------------
    def _fetch_via_akshare(self, code: str, start_date: str) -> pd.DataFrame:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date.replace('-', ''),
            end_date=self.today.replace('-', ''),
            adjust="qfq"
        )
        return self._normalize_columns(df, 'akshare')

    def _fetch_via_baostock(self, code: str, start_date: str) -> pd.DataFrame:
        self._bs_login()

        bs_code = self._to_baostock_code(code)
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,open,high,low,close,volume,pctChg',
            start_date=start_date,
            end_date=self.today,
            frequency='d',
            adjustflag='2'   # 前复权
        )

        if rs is None or rs.error_code != '0':
            raise RuntimeError(f"Baostock 查询失败: {rs.error_msg if rs else 'rs is None'}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        return self._normalize_columns(df, 'baostock')

    # ------------------------------------------------------------------
    # 私有方法：工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_code(code: str) -> str:
        """去掉 sz/sh 前缀和后缀，补足6位，统一为纯数字代码。"""
        code = code.lstrip('sz').lstrip('sh').lstrip('SZ').lstrip('SH')
        code = code.rstrip('.sz').rstrip('.sh').rstrip('.SZ').rstrip('.SH')
        return code.zfill(6)

    @staticmethod
    def _to_baostock_code(code: str) -> str:
        """将纯数字代码转为 Baostock 格式 (sh.600000 / sz.000001)。"""
        # 补足6位，不去掉前导零
        code = code.zfill(6)
        if code.startswith(('6', '9')):
            return f"sh.{code}"
        else:
            return f"sz.{code}"

    @staticmethod
    def _period_to_start_date(period: str) -> str:
        """返回 YYYY-MM-DD 格式，AKShare 和 Baostock 都兼容。"""
        day_map = {'1年': 365, '6个月': 183, '3个月': 91, '1个月': 30, '1周': 7}
        days = day_map.get(period, 365)
        return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
