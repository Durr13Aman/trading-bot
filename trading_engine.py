import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import os
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, symbols: List[str] = None, symbols_file: str = "symbols.txt"):
        self.symbols_file = symbols_file
        self.symbols = self.load_symbols() if symbols is None else symbols
        self.position_size_pct = 10.0  # 10% of original capital per trade
        self.stop_loss_pct = 0.5  # 0.5% below entry
        self.support_tolerance = 0.2  # 0.2% tolerance for support
        self.support_resistance_levels = {}
        self.last_analysis_time = None
        
        # Note: Active trades are now managed by PortfolioManager
        # This engine only generates signals
        
    def load_symbols(self) -> List[str]:
        """Load symbols from text file"""
        try:
            if os.path.exists(self.symbols_file):
                with open(self.symbols_file, 'r') as f:
                    symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
                    if symbols:
                        logger.info(f"Loaded {len(symbols)} symbols from {self.symbols_file}")
                        return symbols
        except Exception as e:
            logger.error(f"Error loading symbols from file: {e}")
        
        # Return default symbols if file doesn't exist or is empty
        default_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        logger.info(f"Using default symbols: {default_symbols}")
        self.save_symbols(default_symbols)
        return default_symbols
    
    def save_symbols(self, symbols: List[str] = None) -> bool:
        """Save symbols to text file"""
        try:
            symbols_to_save = symbols or self.symbols
            with open(self.symbols_file, 'w') as f:
                for symbol in symbols_to_save:
                    f.write(f"{symbol.upper()}\n")
            logger.info(f"Saved {len(symbols_to_save)} symbols to {self.symbols_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving symbols to file: {e}")
            return False
    
    def add_symbol(self, symbol: str) -> bool:
        """Add a new symbol to monitor and save to file"""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            if self.save_symbols():
                logger.info(f"Added {symbol} to monitoring list")
                return True
        return False
            
    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from monitoring and save to file"""
        symbol = symbol.upper()
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            # Clean up any existing data in support/resistance levels
            if symbol in self.support_resistance_levels:
                del self.support_resistance_levels[symbol]
            
            if self.save_symbols():
                logger.info(f"Removed {symbol} from monitoring list")
                return True
        return False
            
    def get_symbols(self) -> List[str]:
        """Get current list of symbols being monitored"""
        return self.symbols.copy()
        
    def fetch_data(self, symbol: str, period='1y', interval='1h') -> Optional[pd.DataFrame]:
        """Fetch hourly data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                logger.error(f"No data found for {symbol}")
                return None
            logger.info(f"Fetched {len(data)} hours of data for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def detect_support_resistance(self, data: pd.DataFrame, lookback=100) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels using swing highs and lows"""
        window = 3
        
        # Find swing highs (resistance)
        resistance_levels = []
        for i in range(window, len(data) - window):
            current_high = data['High'].iloc[i]
            is_swing_high = True
            
            for j in range(1, window + 1):
                if (current_high <= data['High'].iloc[i-j] or 
                    current_high <= data['High'].iloc[i+j]):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                resistance_levels.append(current_high)
        
        # Find swing lows (support)
        support_levels = []
        for i in range(window, len(data) - window):
            current_low = data['Low'].iloc[i]
            is_swing_low = True
            
            for j in range(1, window + 1):
                if (current_low >= data['Low'].iloc[i-j] or 
                    current_low >= data['Low'].iloc[i+j]):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                support_levels.append(current_low)
        
        current_price = data['Close'].iloc[-1]
        
        # Filter levels correctly
        valid_support = [level for level in support_levels if level < current_price]
        valid_resistance = [level for level in resistance_levels if level > current_price]
        
        # Remove duplicates and sort
        support_levels = sorted(list(set([round(level, 2) for level in valid_support])))
        resistance_levels = sorted(list(set([round(level, 2) for level in valid_resistance])))
        
        # Keep most relevant levels
        support_levels = support_levels[-10:] if support_levels else []
        resistance_levels = resistance_levels[:10] if resistance_levels else []
        
        return support_levels, resistance_levels
    
    def is_pin_bar(self, candle: pd.Series) -> bool:
        """Detect bullish pin bar pattern"""
        open_price = candle['Open']
        high_price = candle['High']
        low_price = candle['Low']
        close_price = candle['Close']
        
        body_size = abs(close_price - open_price)
        lower_wick = min(open_price, close_price) - low_price
        upper_wick = high_price - max(open_price, close_price)
        
        total_range = high_price - low_price
        if total_range == 0:
            return False
            
        close_position = (close_price - low_price) / total_range
        
        # Pin bar conditions (relaxed for more signals)
        is_lower_wick_long = lower_wick >= 2 * body_size if body_size > 0 else lower_wick > total_range * 0.4
        is_close_in_upper_area = close_position >= 0.6
        is_body_significant = body_size >= total_range * 0.03
        
        return is_lower_wick_long and is_close_in_upper_area and is_body_significant
    
    def find_nearest_support(self, current_price: float, support_levels: List[float]) -> Optional[float]:
        """Find the nearest support level below current price"""
        valid_supports = [level for level in support_levels if level <= current_price]
        if not valid_supports:
            return None
        return max(valid_supports)
    
    def find_next_resistance(self, current_price: float, resistance_levels: List[float]) -> Optional[float]:
        """Find the next resistance level above current price"""
        valid_resistances = [level for level in resistance_levels if level >= current_price]
        if not valid_resistances:
            return None
        return min(valid_resistances)
    
    def check_entry_signal(self, symbol: str, data: pd.DataFrame, total_capital: float) -> Tuple[bool, str, Dict]:
        """Check if all entry conditions are met"""
        if len(data) < 50:
            return False, "Insufficient data", {}
        
        latest_candle = data.iloc[-1]
        current_price = latest_candle['Close']
        
        # Calculate position size as 10% of original total capital
        position_size = total_capital * (self.position_size_pct / 100)
        
        # Get support/resistance levels
        support_levels, resistance_levels = self.detect_support_resistance(data)
        self.support_resistance_levels[symbol] = {
            'support': support_levels,
            'resistance': resistance_levels,
            'current_price': current_price,
            'timestamp': data.index[-1]
        }
        
        if not support_levels:
            return False, "No support levels identified", {}
        
        # Find nearest support
        nearest_support = self.find_nearest_support(current_price, support_levels)
        if nearest_support is None:
            return False, "No valid support level found", {}
        
        # Check pin bar
        is_pin = self.is_pin_bar(latest_candle)
        if not is_pin:
            return False, "No pin bar pattern", {}
        
        # Check price proximity to support
        support_diff_pct = abs(current_price - nearest_support) / nearest_support * 100
        if support_diff_pct > self.support_tolerance:
            return False, f"Price too far from support: {support_diff_pct:.2f}%", {}
        
        # All conditions met! Generate signal info
        signal_info = {
            'symbol': symbol,
            'entry_price': current_price,
            'support_level': nearest_support,
            'target': self.find_next_resistance(current_price, resistance_levels),
            'stop_loss': current_price * (1 - self.stop_loss_pct / 100),
            'timestamp': data.index[-1],
            'support_diff_pct': support_diff_pct,
            'position_size': position_size,
            'position_size_pct': self.position_size_pct
        }
        
        return True, "Entry signal generated", signal_info
    
    def check_exit_conditions(self, symbol: str, trade_info: Dict, current_data: pd.DataFrame) -> Tuple[bool, str, float]:
        """Check if exit conditions are met for active trade"""
        current_price = current_data['Close'].iloc[-1]
        
        # Check stop loss
        if current_price <= trade_info['stop_loss']:
            return True, "Stop loss triggered", current_price
        
        # Check target
        if trade_info.get('target') and current_price >= trade_info['target']:
            return True, "Target reached", current_price
        
        return False, "Hold position", current_price
    
    def analyze_symbol(self, symbol: str, active_trades: Dict = None, total_capital: float = 100000) -> Dict:
        """Analyze a single symbol and return results"""
        active_trades = active_trades or {}
        
        result = {
            'symbol': symbol,
            'data_fetched': False,
            'has_signal': False,
            'should_exit': False,
            'signal_info': {},
            'exit_info': {},
            'support_resistance': {},
            'current_price': None,
            'pin_bar_detected': False,
            'message': ''
        }
        
        try:
            # Fetch data
            data = self.fetch_data(symbol)
            if data is None:
                result['message'] = "Failed to fetch data"
                return result
                
            result['data_fetched'] = True
            result['current_price'] = data['Close'].iloc[-1]
            
            # Check for exit conditions on active trades first
            if symbol in active_trades:
                should_exit, reason, exit_price = self.check_exit_conditions(
                    symbol, active_trades[symbol], data
                )
                if should_exit:
                    result['should_exit'] = True
                    result['exit_info'] = {
                        'symbol': symbol,
                        'exit_price': exit_price,
                        'exit_reason': reason,
                        'timestamp': data.index[-1]
                    }
                    result['message'] = f"EXIT: {reason}"
                    return result
                else:
                    result['message'] = "Active trade ongoing"
            
            # Check for new entry signals (only if no active trade)
            if symbol not in active_trades:
                has_signal, message, signal_data = self.check_entry_signal(symbol, data, total_capital)
                result['has_signal'] = has_signal
                result['message'] = message
                
                if has_signal:
                    result['signal_info'] = signal_data
                    
            # Always update support/resistance info
            if symbol in self.support_resistance_levels:
                result['support_resistance'] = self.support_resistance_levels[symbol]
                
            # Check if latest candle is a pin bar
            result['pin_bar_detected'] = self.is_pin_bar(data.iloc[-1])
            
        except Exception as e:
            result['message'] = f"Error analyzing {symbol}: {str(e)}"
            logger.error(f"Error analyzing {symbol}: {e}")
            
        return result
    
    def run_analysis(self, active_trades: Dict = None, total_capital: float = 100000) -> Dict:
        """Run analysis on all symbols and return comprehensive results"""
        active_trades = active_trades or {}
        self.last_analysis_time = datetime.now()
        
        results = {
            'timestamp': self.last_analysis_time,
            'symbols_analyzed': [],
            'new_signals_count': 0,
            'exit_signals_count': 0,
            'symbol_results': {},
            'max_concurrent_trades': 10,
            'current_active_trades': len(active_trades)
        }
        
        logger.info("=== Running analysis ===")
        logger.info(f"Current active trades: {len(active_trades)}/10")
        
        for symbol in self.symbols:
            symbol_result = self.analyze_symbol(symbol, active_trades, total_capital)
            results['symbol_results'][symbol] = symbol_result
            results['symbols_analyzed'].append(symbol)
            
            if symbol_result['has_signal']:
                results['new_signals_count'] += 1
                logger.info(f"🟢 NEW SIGNAL: {symbol} - Position size: ₹{symbol_result['signal_info']['position_size']:,.2f} (10% of capital)")
                
            if symbol_result['should_exit']:
                results['exit_signals_count'] += 1
                logger.info(f"🔴 EXIT SIGNAL: {symbol} - {symbol_result['message']}")
        
        logger.info(f"📊 Analysis complete: {results['new_signals_count']} new signals, {results['exit_signals_count']} exit signals")
        return results
    
    def get_status(self) -> Dict:
        """Get current status of the trading engine"""
        return {
            'symbols': self.symbols,
            'support_resistance_levels': self.support_resistance_levels,
            'last_analysis_time': self.last_analysis_time,
            'symbols_count': len(self.symbols)
        }
    
    def get_symbol_data(self, symbol: str, period='5d', interval='1h') -> Optional[pd.DataFrame]:
        """Get recent data for a symbol for charting"""
        return self.fetch_data(symbol, period, interval)

# Example usage
if __name__ == "__main__":
    # Test the engine
    engine = NiftyTradingEngine()
    results = engine.run_analysis()
    print(f"Analysis completed with {results['new_signals_count']} signals")