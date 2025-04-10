# AIMoney - Smart Stock Analysis Platform

AIMoney is a sophisticated stock analysis platform that provides real-time technical analysis and financial insights for stock market investors.

## Features

- Real-time stock data analysis
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Moving average crossover analysis
- Financial metrics and company information
- Interactive charts and visualizations
- Live market updates

## Technical Stack

- Python 3.11+
- Flask web framework
- yfinance for stock data
- Pandas & NumPy for data analysis
- Matplotlib for charting
- Bootstrap for UI
- Gunicorn for production deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aimoney.git
cd aimoney
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Development
```bash
python app.py
```

### Production
```bash
gunicorn app:app
```

## Environment Variables

No API keys are required as the application uses the free yfinance API.

## Deployment

The application is configured for deployment on Render.com. See `render.yaml` for deployment configuration.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- yfinance for providing stock market data
- Technical analysis community for indicator formulas
- Flask community for the web framework 