import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

app = Flask(__name__)

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators with color-coding"""
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Death Cross and Golden Cross
    df['Death_Cross'] = (df['MA50'] < df['MA200']) & (df['MA50'].shift(1) >= df['MA200'].shift(1))
    df['Golden_Cross'] = (df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1))
    
    # Bollinger Bands (20-day, 2 standard deviations)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Stochastic Oscillator (14,3,3)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['Volume']).cumsum()
    
    # Average Directional Index (ADX)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = true_range
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return df

def get_financial_info(ticker):
    """Get financial information for a stock"""
    stock = yf.Ticker(ticker)
    
    try:
        # Get company info
        info = stock.info
        
        # Debug logging for dividend yield
        raw_dividend = info.get('dividendYield', 0)
        print(f"Raw dividend yield from yfinance: {raw_dividend}")
        print(f"Type of dividend yield: {type(raw_dividend)}")
        print(f"All dividend related fields: {[k for k in info.keys() if 'dividend' in k.lower()]}")
        
        # Get historical data for technical indicators
        hist = stock.history(period="1y")
        
        # Calculate technical indicators
        hist = calculate_technical_indicators(hist)
        latest = hist.iloc[-1]
        
        # Extract relevant financial data
        financial_data = {
            'market_cap': f"${info.get('marketCap', 0) / 1e9:.2f}B",
            'pe_ratio': f"{info.get('trailingPE', 0):.2f}",
            'dividend_yield': float(raw_dividend) if raw_dividend is not None else None,  # Handle None case explicitly
            'beta': f"{info.get('beta', 0):.2f}",
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'employees': f"{info.get('fullTimeEmployees', 0):,}",
            'description': info.get('longBusinessSummary', 'No description available.'),
            'website': info.get('website', ''),
            'revenue': f"${info.get('totalRevenue', 0) / 1e9:.2f}B",
            'profit_margin': f"{info.get('profitMargins', 0) * 100:.2f}%",
            'debt_to_equity': f"{info.get('debtToEquity', 0):.2f}",
            'current_ratio': f"{info.get('currentRatio', 0):.2f}",
            'return_on_equity': f"{info.get('returnOnEquity', 0) * 100:.2f}%",
            'return_on_assets': f"{info.get('returnOnAssets', 0) * 100:.2f}%",
            
            # Technical Indicators
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['Signal_Line'],
            'current_price': latest['Close'],
            'bb_upper': latest['BB_upper'],
            'bb_lower': latest['BB_lower'],
            'stoch_k': latest['%K'],
            'stoch_d': latest['%D'],
            'volume': latest['Volume'],
            'avg_volume': hist['Volume'].rolling(window=20).mean().iloc[-1],
            'adx': latest['ADX'],
            'plus_di': latest['Plus_DI'],
            'minus_di': latest['Minus_DI'],
            'atr': latest['ATR'],
            'obv': latest['OBV'],
            'vwap': latest['VWAP']
        }
        
        return financial_data
    except Exception as e:
        print(f"Error getting financial info: {str(e)}")
        return {}

def analyze_stock(ticker, start_date=None, end_date=None, include_indicators=True):
    """Analyze a stock with optional date range and technical indicators"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None, "No data available for this stock."
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Get latest values for indicators
        latest = df.iloc[-1]
        
        # Check for Death Cross
        death_cross_dates = df[df['Death_Cross']].index
        has_death_cross = len(death_cross_dates) > 0
        latest_death_cross = death_cross_dates[-1] if has_death_cross else None
        
        # Check for Golden Cross
        golden_cross_dates = df[df['Golden_Cross']].index
        has_golden_cross = len(golden_cross_dates) > 0
        latest_golden_cross = golden_cross_dates[-1] if has_golden_cross else None
        
        # Check current death cross status (if 50MA is below 200MA)
        current_death_cross = df['MA50'].iloc[-1] < df['MA200'].iloc[-1]
        
        # Create price chart with moving averages
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.plot(df.index, df['MA20'], label='20-day MA', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['MA50'], label='50-day MA', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['MA200'], label='200-day MA', linestyle='--', alpha=0.7)
        
        # Mark Death Cross and Golden Cross points
        if has_death_cross:
            plt.scatter(death_cross_dates, df.loc[death_cross_dates, 'Close'], 
                       color='red', s=100, marker='v', label='Death Cross')
        
        if has_golden_cross:
            plt.scatter(golden_cross_dates, df.loc[golden_cross_dates, 'Close'], 
                       color='green', s=100, marker='^', label='Golden Cross')
        
        plt.title(f'{ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        # Save price chart to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        price_chart = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Create technical indicator charts if requested
        rsi_chart = None
        macd_chart = None
        
        if include_indicators:
            # RSI Chart
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df['RSI'])
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.title('Relative Strength Index (RSI)')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.grid(True)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            rsi_chart = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            # MACD Chart
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df['MACD'], label='MACD')
            plt.plot(df.index, df['Signal_Line'], label='Signal Line')
            plt.title('Moving Average Convergence Divergence (MACD)')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.legend()
            plt.grid(True)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            macd_chart = base64.b64encode(img.getvalue()).decode()
            plt.close()
        
        # Get financial information and update with latest technical indicators
        financial_info = get_financial_info(ticker)
        financial_info.update({
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['Signal_Line'],
            'current_price': latest['Close'],
            'bb_upper': latest['BB_upper'],
            'bb_lower': latest['BB_lower'],
            'stoch_k': latest['%K'],
            'stoch_d': latest['%D'],
            'volume': latest['Volume'],
            'avg_volume': df['Volume'].rolling(window=20).mean().iloc[-1],
            'adx': latest['ADX'],
            'plus_di': latest['Plus_DI'],
            'minus_di': latest['Minus_DI'],
            'atr': latest['ATR'],
            'obv': latest['OBV'],
            'vwap': latest['VWAP']
        })
        
        return {
            'ticker': ticker,
            'company_name': stock.info.get('longName', ticker),
            'price_chart': price_chart,
            'rsi_chart': rsi_chart,
            'macd_chart': macd_chart,
            'financial_info': financial_info,
            'include_indicators': include_indicators,
            'has_death_cross': has_death_cross,
            'latest_death_cross': latest_death_cross.strftime('%Y-%m-%d') if latest_death_cross else None,
            'has_golden_cross': has_golden_cross,
            'latest_golden_cross': latest_golden_cross.strftime('%Y-%m-%d') if latest_golden_cross else None,
            'current_death_cross': current_death_cross
        }, None
        
    except Exception as e:
        return None, str(e)

def analyze_indicator_signals(df):
    """Analyze all technical indicators and return their signals with color-coding"""
    latest = df.iloc[-1]
    signals = {
        'trend_indicators': {
            'name': 'Trend Indicators',
            'signals': [
                {
                    'name': 'Moving Averages',
                    'is_bullish': latest['MA50'] > latest['MA200'],
                    'value': f"50MA: {latest['MA50']:.2f} vs 200MA: {latest['MA200']:.2f}",
                    'description': '50-day MA above 200-day MA is bullish'
                },
                {
                    'name': 'ADX',
                    'is_bullish': latest['ADX'] > 25 and latest['Plus_DI'] > latest['Minus_DI'],
                    'value': f"ADX: {latest['ADX']:.2f}, +DI: {latest['Plus_DI']:.2f}, -DI: {latest['Minus_DI']:.2f}",
                    'description': 'ADX>25 shows strong trend, +DI>-DI is bullish'
                }
            ]
        },
        'momentum_indicators': {
            'name': 'Momentum Indicators',
            'signals': [
                {
                    'name': 'RSI',
                    'is_bullish': 40 <= latest['RSI'] <= 60,
                    'value': f"{latest['RSI']:.2f}",
                    'description': 'RSI between 40-60 is neutral, <30 oversold, >70 overbought'
                },
                {
                    'name': 'MACD',
                    'is_bullish': latest['MACD'] > latest['Signal_Line'],
                    'value': f"MACD: {latest['MACD']:.2f} vs Signal: {latest['Signal_Line']:.2f}",
                    'description': 'MACD above Signal Line is bullish'
                },
                {
                    'name': 'Stochastic',
                    'is_bullish': 20 <= latest['%K'] <= 80 and 20 <= latest['%D'] <= 80,
                    'value': f"%K: {latest['%K']:.2f}, %D: {latest['%D']:.2f}",
                    'description': 'Values between 20-80 indicate healthy trend'
                },
                {
                    'name': 'MFI',
                    'is_bullish': 40 <= latest['MFI'] <= 60,
                    'value': f"{latest['MFI']:.2f}",
                    'description': 'MFI between 40-60 is neutral, <20 oversold, >80 overbought'
                }
            ]
        },
        'volatility_indicators': {
            'name': 'Volatility Indicators',
            'signals': [
                {
                    'name': 'Bollinger Bands',
                    'is_bullish': latest['Close'] > latest['BB_middle'],
                    'value': f"Price: {latest['Close']:.2f}, Middle: {latest['BB_middle']:.2f}",
                    'description': 'Price above middle band is bullish'
                },
                {
                    'name': 'ATR',
                    'is_bullish': None,  # ATR doesn't indicate direction
                    'value': f"{latest['ATR']:.2f}",
                    'description': 'Measures volatility, higher values = more volatile'
                }
            ]
        },
        'volume_indicators': {
            'name': 'Volume Indicators',
            'signals': [
                {
                    'name': 'OBV',
                    'is_bullish': df['OBV'].diff().iloc[-1] > 0,
                    'value': f"OBV Change: {df['OBV'].diff().iloc[-1]:.0f}",
                    'description': 'Rising OBV confirms price trend'
                },
                {
                    'name': 'VWAP',
                    'is_bullish': latest['Close'] > latest['VWAP'],
                    'value': f"Price vs VWAP: {latest['Close']:.2f} vs {latest['VWAP']:.2f}",
                    'description': 'Price above VWAP is bullish'
                }
            ]
        }
    }
    return signals

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_stock_route():
    try:
        symbol = request.form.get('symbol')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        if not symbol:
            return render_template('index.html', error='Please enter a stock symbol')
        
        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            result, error = analyze_stock(symbol, start_date, end_date)
            if error:
                return jsonify({'error': error})
            return jsonify(result)
        
        # Regular form submission
        result, error = analyze_stock(symbol, start_date, end_date)
        
        if error:
            return render_template('index.html', error=error)
        
        return render_template('index.html', stock_data=result)
        
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/live', methods=['POST'])
def get_live_data():
    try:
        symbol = request.form.get('symbol')
        if not symbol:
            return jsonify({'error': 'Please enter a stock symbol'})
        
        # Get stock data using yfinance
        stock = yf.Ticker(symbol)
        
        try:
            # Get real-time quote information
            info = stock.info
            if not info:
                return jsonify({'error': 'Unable to fetch stock data'})
            
            # Get current price and changes
            current_price = info.get('regularMarketPrice')
            price_change = info.get('regularMarketChange')
            current_volume = info.get('regularMarketVolume')
            previous_close = info.get('regularMarketPreviousClose')
            
            if not current_price or not previous_close:
                return jsonify({'error': 'No price data available'})
            
            # Get daily data for technical indicators
            daily_data = stock.history(period='5d')
            if daily_data.empty:
                return jsonify({'error': 'No historical data available'})
            
            # Calculate technical indicators
            daily_data = calculate_technical_indicators(daily_data)
            latest = daily_data.iloc[-1]
            
            # Check if market is open (US market hours)
            now = datetime.now()
            is_market_open = (
                now.weekday() < 5 and  # Monday to Friday
                9 <= now.hour < 16 or  # Between 9 AM and 4 PM
                (now.hour == 16 and now.minute == 0)  # Include 4:00 PM
            )
            
            # Helper function to safely convert numpy/pandas values to Python native types
            def safe_float(value):
                try:
                    if pd.isna(value) or np.isnan(value):
                        return None
                    return float(value)
                except:
                    return None
            
            response_data = {
                'success': True,
                'current_time': now.strftime('%H:%M:%S'),
                'price': safe_float(current_price),
                'previousClose': safe_float(previous_close),
                'price_change': safe_float(price_change),
                'volume': int(current_volume) if current_volume else 0,
                'macd': safe_float(latest.get('MACD')),
                'signal_line': safe_float(latest.get('Signal_Line')),
                'is_market_open': is_market_open
            }
            
            # Remove any None values from the response
            response_data = {k: v for k, v in response_data.items() if v is not None}
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error in live data processing: {str(e)}")
            return jsonify({'error': 'Error processing live data'})
            
    except Exception as e:
        print(f"Critical error in live data: {str(e)}")
        return jsonify({'error': 'Critical error fetching data'})

def generate_charts(df, symbol):
    """Generate all technical analysis charts with proper color-coding"""
    charts = {}
    
    # Price and Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Price', color='blue')
    plt.plot(df.index, df['MA20'], label='20 MA', color='orange', alpha=0.7)
    plt.plot(df.index, df['MA50'], label='50 MA', color='green', alpha=0.7)
    plt.plot(df.index, df['MA200'], label='200 MA', color='red', alpha=0.7)
    plt.title(f'{symbol} Price and Moving Averages')
    plt.legend()
    plt.grid(True)
    charts['price_ma'] = get_chart_data()
    plt.close()
    
    # RSI
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['RSI'], label='RSI', color='purple')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.fill_between(df.index, 70, 100, color='red', alpha=0.1)
    plt.fill_between(df.index, 0, 30, color='green', alpha=0.1)
    plt.title('RSI (14)')
    plt.legend()
    plt.grid(True)
    charts['rsi'] = get_chart_data()
    plt.close()
    
    # MACD
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['Signal_Line'], label='Signal', color='red')
    plt.bar(df.index, df['MACD_Histogram'], label='Histogram', color=np.where(df['MACD_Histogram'] >= 0, 'green', 'red'))
    plt.title('MACD')
    plt.legend()
    plt.grid(True)
    charts['macd'] = get_chart_data()
    plt.close()
    
    # Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Price', color='blue')
    plt.plot(df.index, df['BB_upper'], label='Upper BB', color='red', alpha=0.7)
    plt.plot(df.index, df['BB_middle'], label='Middle BB', color='green', alpha=0.7)
    plt.plot(df.index, df['BB_lower'], label='Lower BB', color='red', alpha=0.7)
    plt.fill_between(df.index, df['BB_upper'], df['BB_lower'], color='gray', alpha=0.1)
    plt.title('Bollinger Bands')
    plt.legend()
    plt.grid(True)
    charts['bollinger'] = get_chart_data()
    plt.close()
    
    # Stochastic Oscillator
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['%K'], label='%K', color='blue')
    plt.plot(df.index, df['%D'], label='%D', color='red')
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    plt.fill_between(df.index, 80, 100, color='red', alpha=0.1)
    plt.fill_between(df.index, 0, 20, color='green', alpha=0.1)
    plt.title('Stochastic Oscillator')
    plt.legend()
    plt.grid(True)
    charts['stochastic'] = get_chart_data()
    plt.close()
    
    # Volume and OBV
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(df.index, df['Close'], label='Price', color='blue')
    ax1.set_title(f'{symbol} Price and Volume')
    ax1.legend()
    ax1.grid(True)
    
    ax2.bar(df.index, df['Volume'], label='Volume', color=np.where(df['Close'] >= df['Close'].shift(1), 'green', 'red'))
    ax2.plot(df.index, df['OBV'], label='OBV', color='purple')
    ax2.set_title('Volume and OBV')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    charts['volume_obv'] = get_chart_data()
    plt.close()
    
    return charts

if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=10000, debug=False) 