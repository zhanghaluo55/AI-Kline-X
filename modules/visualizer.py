import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Bar, Grid
from pyecharts.commons.utils import JsCode

class Visualizer:
    """
    可视化类，负责生成K线图和各种技术指标图表
    """
    
    def __init__(self):
        # 设置matplotlib中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    def _get_stock_name(self, stock_code: str) -> str:
        """获取股票名称，AKShare 优先，Baostock 备选。"""
        code = stock_code.lstrip('sz').lstrip('sh').lstrip('SZ').lstrip('SH').rstrip(
            '.sz').rstrip('.sh').rstrip('.SZ').rstrip('.SH').zfill(6)
        bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"

        # 尝试 Baostock
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

        # 尝试 AKShare
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
    
    def create_charts(self, stock_data, indicators, stock_code, save_path):
        """
        创建K线图和技术指标图表
        
        参数:
            stock_data (pandas.DataFrame): 股票历史数据
            indicators (dict): 技术指标数据
            stock_code (str): 股票代码
            save_path (str): 保存路径
            
        返回:
            str: 图表保存路径
        """
        if stock_data.empty:
            return ""
        
        # 获取股票名称（AKShare 优先，Baostock 备选）
        stock_name = self._get_stock_name(stock_code)

        
        # 创建保存目录
        chart_dir = os.path.join(save_path, 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 使用matplotlib创建图表
        self._create_matplotlib_charts(stock_data, indicators, stock_code, stock_name, chart_dir)
        
        # 使用pyecharts创建交互式图表
        self._create_pyecharts_charts(stock_data, indicators, stock_code, stock_name, chart_dir)
        
        return chart_dir
    
    def _create_matplotlib_charts(self, stock_data, indicators, stock_code, stock_name, save_path):
        """
        使用matplotlib创建图表
        """
        # 创建一个大图，包含多个子图
        fig = plt.figure(figsize=(16, 12))
        
        # 设置网格
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # 添加K线图和移动平均线
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title(f"{stock_name}({stock_code}) K线图与技术指标")
        
        # 绘制K线图
        for i in range(len(stock_data)):
            # 绘制蜡烛图
            if stock_data['close'].iloc[i] >= stock_data['open'].iloc[i]:
                # 收盘价大于等于开盘价，为阳线
                color = 'red'
            else:
                # 收盘价小于开盘价，为阴线
                color = 'green'
            
            # 绘制实体部分
            ax1.plot([i, i], [stock_data['open'].iloc[i], stock_data['close'].iloc[i]], 
                     color=color, linewidth=8, solid_capstyle='butt')
            # 绘制上下影线
            ax1.plot([i, i], [stock_data['low'].iloc[i], stock_data['high'].iloc[i]], 
                     color=color, linewidth=1)
        
        # 绘制移动平均线
        ax1.plot(indicators['MA5'], label='MA5', linewidth=1)
        ax1.plot(indicators['MA10'], label='MA10', linewidth=1)
        ax1.plot(indicators['MA20'], label='MA20', linewidth=1)
        ax1.plot(indicators['MA30'], label='MA30', linewidth=1)
        
        # 绘制布林带
        ax1.plot(indicators['BOLL_upper'], label='BOLL上轨', linestyle='--', linewidth=1)
        ax1.plot(indicators['BOLL_middle'], label='BOLL中轨', linestyle='-', linewidth=1)
        ax1.plot(indicators['BOLL_lower'], label='BOLL下轨', linestyle='--', linewidth=1)
        
        # 设置x轴刻度
        ax1.set_xticks(range(0, len(stock_data), len(stock_data) // 10))
        ax1.set_xticklabels([d.strftime('%Y-%m-%d') for d in stock_data['date'].iloc[::len(stock_data) // 10]])
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 添加成交量图
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_title("成交量")
        for i in range(len(stock_data)):
            if stock_data['close'].iloc[i] >= stock_data['open'].iloc[i]:
                color = 'red'
            else:
                color = 'green'
            ax2.bar(i, stock_data['volume'].iloc[i], color=color, width=0.8)
        
        # 绘制成交量移动平均线
        ax2.plot(indicators['volume_ma5'], label='Volume MA5', color='blue', linewidth=1)
        ax2.plot(indicators['volume_ma10'], label='Volume MA10', color='orange', linewidth=1)
        ax2.legend(loc='best')
        ax2.grid(True)
        
        # 添加MACD图
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_title("MACD")
        ax3.plot(indicators['MACD'], label='MACD', color='blue', linewidth=1)
        ax3.plot(indicators['MACD_signal'], label='Signal', color='orange', linewidth=1)
        
        # 绘制MACD柱状图
        for i in range(len(indicators['MACD_hist'])):
            if indicators['MACD_hist'].iloc[i] >= 0:
                color = 'red'
            else:
                color = 'green'
            ax3.bar(i, indicators['MACD_hist'].iloc[i], color=color, width=0.8)
        
        ax3.legend(loc='best')
        ax3.grid(True)
        
        # 添加KDJ图
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.set_title("KDJ")
        ax4.plot(indicators['K'], label='K', color='blue', linewidth=1)
        ax4.plot(indicators['D'], label='D', color='orange', linewidth=1)
        ax4.plot(indicators['J'], label='J', color='green', linewidth=1)
        ax4.axhline(y=80, color='r', linestyle='--', alpha=0.3)
        ax4.axhline(y=20, color='g', linestyle='--', alpha=0.3)
        ax4.legend(loc='best')
        ax4.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(save_path, f"{stock_code}_technical_analysis.png"), dpi=300)
        plt.close()
    
    def _create_pyecharts_charts(self, stock_data, indicators, stock_code, stock_name, save_path):
        """
        使用pyecharts创建交互式图表
        """
        # 准备数据
        dates = stock_data['date'].dt.strftime('%Y-%m-%d').tolist()
        k_data = [[float(stock_data['open'].iloc[i]), 
                  float(stock_data['close'].iloc[i]), 
                  float(stock_data['low'].iloc[i]), 
                  float(stock_data['high'].iloc[i])] for i in range(len(stock_data))]
        
        # 创建K线图
        kline = Kline()
        kline.add_xaxis(dates)
        kline.add_yaxis(
            "K线",
            k_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",
                color0="#14b143",
                border_color="#ef232a",
                border_color0="#14b143",
            ),
        )
        
        # K线图设置标题
        kline.set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{stock_name}({stock_code}) K线图与成交量分析", 
                pos_left="center",
                padding=[10, 0, 0, 0],
                pos_top="1%"
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
            ],
            legend_opts=opts.LegendOpts(
                pos_bottom="0%",  # 将图例放在底部
                pos_left="center",  # 图例居中
                orient="horizontal",  # 水平排列
                item_gap=20  # 图例项之间的间隔
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                orient="horizontal",
                pos_right="5%",
                pos_top="top",
                feature={
                    "saveAsImage": {},
                    "dataZoom": {},
                    "dataView": {},
                    "restore": {},
                }
            ),
        )
        
        # 创建MA线
        line = Line()
        line.add_xaxis(dates)
        line.add_yaxis("MA5", indicators['MA5'].round(2).tolist(), is_smooth=True, is_symbol_show=False, 
                      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
        line.add_yaxis("MA10", indicators['MA10'].round(2).tolist(), is_smooth=True, is_symbol_show=False, 
                      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
        line.add_yaxis("MA20", indicators['MA20'].round(2).tolist(), is_smooth=True, is_symbol_show=False, 
                      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
        line.add_yaxis("MA30", indicators['MA30'].round(2).tolist(), is_smooth=True, is_symbol_show=False, 
                      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
        
        # 将线叠加到K线图上
        overlap_kline = kline.overlap(line)
        
        # 创建成交量图
        bar = Bar()
        bar.add_xaxis(dates)
        bar.add_yaxis(
            "成交量",
            stock_data['volume'].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                    function(params) {
                        var colorList;
                        if (params.data >= 0) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                )
            ),
        )
        
        # 成交量图设置
        bar.set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitline_opts=opts.SplitLineOpts(is_show=True),
                name="成交量",  # 在Y轴上标明这是成交量
                name_location="middle",  # 名称位置
                name_gap=40,  # 与轴的距离
                name_rotate=90,  # 旋转角度
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
        
        # 创建网格布局
        grid = Grid(init_opts=opts.InitOpts(
            width="100%",
            height="700px",
            page_title=f"AI看线 - {stock_name}({stock_code}) 技术分析"
        ))
        
        # 添加K线图和成交量图到网格
        grid.add(overlap_kline, grid_opts=opts.GridOpts(
            pos_left="10%", 
            pos_right="8%", 
            pos_top="10%",  # 距离顶部的距离，为标题留出空间
            height="60%"  # 占整体高度的比例
        ))
        
        grid.add(bar, grid_opts=opts.GridOpts(
            pos_left="10%", 
            pos_right="8%", 
            pos_top="75%",  # 距离顶部的距离，确保与K线图有足够间隔
            height="20%"  # 占整体高度的比例
        ))
        
        # 保存图表
        grid.render(os.path.join(save_path, f"{stock_code}_interactive_chart.html"))
        
        # 读取生成的HTML文件，添加自适应大小的脚本
        html_path = os.path.join(save_path, f"{stock_code}_interactive_chart.html")
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # 添加自定义样式和Meta标签
            meta_tags = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            margin: 0; 
            padding: 0; 
            font-family: "Microsoft YaHei", Arial, sans-serif; 
        }
        .container {
            width: 100%;
            height: 100%;
            padding: 0;
            margin: 0;
            overflow: hidden;
        }
        /* 标题样式优化 */
        .title-text {
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 15px 0 !important;
            margin-bottom: 15px !important;
        }
        /* 图例样式优化 */
        .legend {
            padding-top: 15px !important;
            display: flex !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
        }
        .legend-item {
            margin: 0 10px !important;
            display: inline-flex !important;
            align-items: center !important;
        }
        /* 确保各种尺寸屏幕上不出现文字重叠 */
        @media (max-width: 768px) {
            .title-text {
                font-size: 14px !important;
            }
            .legend-item {
                margin: 0 5px !important;
            }
        }
    </style>
"""
            # 在head标签后插入
            html_content = html_content.replace('<head>', '<head>\n' + meta_tags)
            
            # 添加初始化完成后的图表调整脚本
            adjust_script = """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 在页面加载完成后执行额外的调整
            setTimeout(function() {
                // 处理标题元素
                var titleElements = document.querySelectorAll('.title');
                titleElements.forEach(function(el) {
                    el.classList.add('title-text');
                });
                
                // 处理图例元素
                var legendElements = document.querySelectorAll('.legend');
                legendElements.forEach(function(el) {
                    el.style.paddingTop = '15px';
                });
                
                // 处理图例项
                var legendItems = document.querySelectorAll('.legend-item');
                legendItems.forEach(function(el) {
                    el.style.margin = '0 10px';
                });
            }, 500);
        });
    </script>
"""
            # 在</body>标签之前插入
            html_content = html_content.replace('</body>', adjust_script + '</body>')
            
            # 写回文件
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
            print(f"修改HTML文件时出错: {e}")