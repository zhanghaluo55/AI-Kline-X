import os
import json

from flask import Flask, render_template, request, jsonify, send_from_directory
from modules.data_fetcher import StockDataFetcher
from modules.technical_analyzer import TechnicalAnalyzer
from modules.visualizer import Visualizer
from modules.ai_analyzer import AIAnalyzer
from dotenv import load_dotenv
import matplotlib
# 设置 matplotlib 使用非 GUI 后端以避免 tkinter 依赖
matplotlib.use('Agg')

# 加载环境变量
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# 确保输出目录存在
os.makedirs('./output', exist_ok=True)
os.makedirs('./output/charts', exist_ok=True)

# 初始化各模块
data_fetcher = StockDataFetcher()
technical_analyzer = TechnicalAnalyzer()
visualizer = Visualizer()
ai_analyzer = AIAnalyzer()

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """分析股票"""
    data = request.form
    stock_code = data.get('stock_code')
    period = data.get('period', '1年')
    save_path = './output'
    
    if not stock_code:
        return jsonify({'error': '请输入股票代码'}), 400
    
    try:
        # 获取股票数据
        stock_data = data_fetcher.fetch_stock_data(stock_code, period)
        
        if stock_data.empty:
            return jsonify({'error': f'未找到股票 {stock_code} 的数据'}), 404
        
        # 获取财务和新闻数据
        financial_data = data_fetcher.fetch_financial_data(stock_code)
        news_data = data_fetcher.fetch_news_data(stock_code)
        
        # 计算技术指标
        indicators = technical_analyzer.calculate_indicators(stock_data)
        
        # 生成可视化图表
        chart_path = visualizer.create_charts(stock_data, indicators, stock_code, save_path)
        
        # AI分析预测
        analysis_result = ai_analyzer.analyze(stock_data, indicators, financial_data, news_data, stock_code, save_path)
        
        # 保存分析结果
        result_path = os.path.join(save_path, f"{stock_code}_analysis_result.txt")
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(analysis_result)
        
        # 准备返回数据
        chart_files = []
        for file in os.listdir(os.path.join(save_path, 'charts')):
            if file.startswith(stock_code) and (file.endswith('.png') or file.endswith('.html')):
                chart_files.append(file)
        
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'charts': chart_files,
            'analysis_result': analysis_result
        })
    
    except Exception as e:
        return jsonify({'error': f'分析过程中出错: {str(e)}'}), 500

@app.route('/output/charts/<path:filename>')
def serve_chart(filename):
    """提供图表文件"""
    return send_from_directory('output/charts', filename)

@app.route('/stock_info/<stock_code>')
def get_stock_info(stock_code):
    """获取股票基本信息（双数据源）"""
    # 尝试 Baostock（无需联网的 HTTP API，外网可用）
    try:
        import baostock as bs
        code = stock_code.lstrip('sz').lstrip('sh').lstrip('SZ').lstrip('SH').zfill(6)
        bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"
        bs.login()
        rs = bs.query_stock_basic(code=bs_code)
        data = rs.get_data()
        bs.logout()
        if not data.empty:
            info_dict = {
                '股票代码': data['code'].values[0],
                '股票名称': data['code_name'].values[0],
                '上市日期': data.get('ipoDate', pd.Series([''])).values[0],
                '类型': data['type'].values[0],
            }
            return jsonify({'success': True, 'data': info_dict})
    except Exception:
        pass

    # 备选 AKShare
    try:
        import akshare as ak
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        if not stock_info.empty:
            info_dict = {
                row['item']: row['value']
                for _, row in stock_info.iterrows()
            }
            return jsonify({'success': True, 'data': info_dict})
    except Exception as e:
        return jsonify({'error': f'获取股票信息时出错: {str(e)}'}), 500

    return jsonify({'error': f'未找到股票 {stock_code} 的信息'}), 404

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('FLASK_PORT', 5005)))
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port, threaded=False)