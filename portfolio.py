import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, portfolio_file: str = "portfolio.txt"):
        self.portfolio_file = portfolio_file
        self.total_capital = 0
        self.deployed_amount = 0
        self.available_amount = 0
        self.trades_history = []
        self.active_trades = {}
        self.load_portfolio()
    
    def load_portfolio(self) -> bool:
        """Load portfolio data from file"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.total_capital = data.get('total_capital', 100000)
                    self.deployed_amount = data.get('deployed_amount', 0)
                    self.available_amount = data.get('available_amount', 100000)
                    self.trades_history = data.get('trades_history', [])
                    self.active_trades = data.get('active_trades', {})
                    
                logger.info(f"Portfolio loaded: Total=₹{self.total_capital:,.2f}, Available=₹{self.available_amount:,.2f}, Deployed=₹{self.deployed_amount:,.2f}")
                return True
            else:
                # Initialize with default values
                self.total_capital = 100000
                self.deployed_amount = 0
                self.available_amount = 100000
                self.trades_history = []
                self.active_trades = {}
                self.save_portfolio()
                logger.info("Created new portfolio with default values")
                return True
                
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            # Set default values on error
            self.total_capital = 100000
            self.deployed_amount = 0
            self.available_amount = 100000
            self.trades_history = []
            self.active_trades = {}
            return False
    
    def save_portfolio(self) -> bool:
        """Save portfolio data to file"""
        try:
            data = {
                'total_capital': self.total_capital,
                'deployed_amount': self.deployed_amount,
                'available_amount': self.available_amount,
                'trades_history': self.trades_history,
                'active_trades': self.active_trades,
                'last_updated': datetime.now().isoformat(),
                'capital_utilization_pct': (self.deployed_amount / self.total_capital * 100) if self.total_capital > 0 else 0
            }
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Portfolio saved: Total=₹{self.total_capital:,.2f}, Available=₹{self.available_amount:,.2f}, Deployed=₹{self.deployed_amount:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return False
    
    def add_capital(self, amount: float) -> bool:
        """Add capital to the portfolio"""
        if amount <= 0:
            return False
            
        self.total_capital += amount
        self.available_amount += amount
        
        # Log the transaction
        transaction = {
            'type': 'capital_addition',
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'total_capital_after': self.total_capital,
            'available_amount_after': self.available_amount
        }
        self.trades_history.append(transaction)
        
        return self.save_portfolio()
    
    def remove_capital(self, amount: float) -> bool:
        """Remove capital from the portfolio (only if available)"""
        if amount <= 0 or amount > self.available_amount:
            return False
            
        self.total_capital -= amount
        self.available_amount -= amount
        
        # Log the transaction
        transaction = {
            'type': 'capital_removal',
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'total_capital_after': self.total_capital,
            'available_amount_after': self.available_amount
        }
        self.trades_history.append(transaction)
        
        return self.save_portfolio()
    
    def set_total_capital(self, new_total: float) -> bool:
        """Set new total capital (reset portfolio)"""
        if new_total <= 0:
            return False
            
        old_total = self.total_capital
        self.total_capital = new_total
        self.available_amount = new_total - self.deployed_amount
        
        # Ensure available amount is not negative
        if self.available_amount < 0:
            self.available_amount = 0
            self.deployed_amount = new_total
        
        # Log the transaction
        transaction = {
            'type': 'capital_reset',
            'old_total': old_total,
            'new_total': new_total,
            'timestamp': datetime.now().isoformat(),
            'available_amount_after': self.available_amount,
            'deployed_amount': self.deployed_amount
        }
        self.trades_history.append(transaction)
        
        return self.save_portfolio()
    
    def open_trade(self, symbol: str, entry_price: float, amount: float, trade_details: Dict) -> bool:
        """Open a new trade and deploy capital"""
        if amount > self.available_amount:
            logger.error(f"Insufficient funds: Need ₹{amount}, Available ₹{self.available_amount}")
            return False
        
        # Deploy capital
        self.deployed_amount += amount
        self.available_amount -= amount
        
        # Store trade details
        trade_info = {
            'symbol': symbol,
            'entry_price': entry_price,
            'amount_deployed': amount,
            'entry_time': datetime.now().isoformat(),
            'trade_details': trade_details,
            'status': 'active'
        }
        
        self.active_trades[symbol] = trade_info
        
        # Log the transaction
        transaction = {
            'type': 'trade_opened',
            'symbol': symbol,
            'entry_price': entry_price,
            'amount_deployed': amount,
            'timestamp': datetime.now().isoformat(),
            'available_amount_after': self.available_amount,
            'deployed_amount_after': self.deployed_amount
        }
        self.trades_history.append(transaction)
        
        logger.info(f"Trade opened: {symbol} at ₹{entry_price} with ₹{amount}")
        return self.save_portfolio()
    
    def close_trade(self, symbol: str, exit_price: float, exit_reason: str = "Manual") -> bool:
        """Close a trade and calculate P&L"""
        if symbol not in self.active_trades:
            logger.error(f"No active trade found for {symbol}")
            return False
        
        trade = self.active_trades[symbol]
        amount_deployed = trade['amount_deployed']
        entry_price = trade['entry_price']
        
        # Calculate P&L
        price_change_pct = (exit_price - entry_price) / entry_price
        pnl_amount = amount_deployed * price_change_pct
        final_amount = amount_deployed + pnl_amount
        
        # Release capital back to available amount
        self.deployed_amount -= amount_deployed
        self.available_amount += final_amount
        self.total_capital += pnl_amount  # Add/subtract the profit/loss to total capital
        
        # Log the closed trade
        closed_trade = {
            'type': 'trade_closed',
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount_deployed': amount_deployed,
            'pnl_amount': pnl_amount,
            'pnl_percentage': price_change_pct * 100,
            'final_amount': final_amount,
            'exit_reason': exit_reason,
            'entry_time': trade['entry_time'],
            'exit_time': datetime.now().isoformat(),
            'total_capital_after': self.total_capital,
            'available_amount_after': self.available_amount,
            'deployed_amount_after': self.deployed_amount
        }
        
        self.trades_history.append(closed_trade)
        
        # Remove from active trades
        del self.active_trades[symbol]
        
        logger.info(f"Trade closed: {symbol} | P&L: ₹{pnl_amount:.2f} ({price_change_pct*100:.2f}%)")
        return self.save_portfolio()
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        utilization_pct = (self.deployed_amount / self.total_capital * 100) if self.total_capital > 0 else 0
        
        return {
            'total_capital': self.total_capital,
            'available_amount': self.available_amount,
            'deployed_amount': self.deployed_amount,
            'capital_utilization_pct': utilization_pct,
            'active_trades_count': len(self.active_trades),
            'total_trades_history': len([t for t in self.trades_history if t.get('type') in ['trade_opened', 'trade_closed']])
        }
    
    def get_max_concurrent_trades(self) -> int:
        """Get maximum number of concurrent trades allowed (10% each = max 10 trades)"""
        return 10
    
    def get_remaining_trade_slots(self) -> int:
        """Get remaining trade slots available"""
        return self.get_max_concurrent_trades() - len(self.active_trades)
    
    def can_open_new_trade(self) -> bool:
        """Check if we can open a new trade (less than 10 active trades)"""
        return len(self.active_trades) < self.get_max_concurrent_trades()
    
    def get_position_size_for_new_trade(self) -> float:
        """Get position size for new trade (10% of original total capital)"""
        # Use current total capital as base for 10% calculation
        return self.total_capital * 0.10  # 10% of current total capital
    
    def get_capital_utilization_info(self) -> Dict:
        """Get detailed capital utilization information"""
        max_trades = self.get_max_concurrent_trades()
        current_trades = len(self.active_trades)
        position_size = self.get_position_size_for_new_trade()
        
        return {
            'max_concurrent_trades': max_trades,
            'current_active_trades': current_trades,
            'remaining_trade_slots': max_trades - current_trades,
            'position_size_per_trade': position_size,
            'total_potential_deployment': max_trades * position_size,
            'current_deployment': sum([trade['amount_deployed'] for trade in self.active_trades.values()]),
            'remaining_capital_for_trades': (max_trades - current_trades) * position_size
        }
    
    def get_active_trades(self) -> Dict:
        """Get all active trades"""
        return self.active_trades.copy()
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history (most recent first)"""
        return self.trades_history[-limit:][::-1]  # Last 'limit' items, reversed
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        closed_trades = [t for t in self.trades_history if t.get('type') == 'trade_closed']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        pnl_amounts = [t['pnl_amount'] for t in closed_trades]
        winning_trades = len([pnl for pnl in pnl_amounts if pnl > 0])
        losing_trades = len([pnl for pnl in pnl_amounts if pnl < 0])
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / len(closed_trades) * 100) if closed_trades else 0,
            'total_pnl': sum(pnl_amounts),
            'avg_pnl_per_trade': sum(pnl_amounts) / len(closed_trades) if closed_trades else 0,
            'best_trade': max(pnl_amounts) if pnl_amounts else 0,
            'worst_trade': min(pnl_amounts) if pnl_amounts else 0
        }
    
    def update_trade_with_current_price(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Update trade with current price for unrealized P&L calculation"""
        if symbol not in self.active_trades:
            return None
        
        trade = self.active_trades[symbol]
        entry_price = trade['entry_price']
        amount_deployed = trade['amount_deployed']
        
        # Calculate unrealized P&L
        price_change_pct = (current_price - entry_price) / entry_price
        unrealized_pnl = amount_deployed * price_change_pct
        
        return {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': current_price,
            'amount_deployed': amount_deployed,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': price_change_pct * 100,
            'entry_time': trade['entry_time']
        }
    
    def get_max_concurrent_trades(self) -> int:
        """Get maximum number of concurrent trades allowed (10% each = max 10 trades)"""
        return 10
    
    def get_remaining_trade_slots(self) -> int:
        """Get remaining trade slots available"""
        return self.get_max_concurrent_trades() - len(self.active_trades)
    
    def can_open_new_trade(self) -> bool:
        """Check if we can open a new trade (less than 10 active trades)"""
        return len(self.active_trades) < self.get_max_concurrent_trades()
    
    def get_position_size_for_new_trade(self) -> float:
        """Get position size for new trade (10% of original total capital)"""
        # Get the original total capital from the first capital addition or initial setup
        original_capital = self.total_capital
        
        # If we have trade history, find the initial capital
        capital_additions = [t for t in self.trades_history if t.get('type') == 'capital_addition']
        capital_resets = [t for t in self.trades_history if t.get('type') == 'capital_reset']
        
        if capital_resets:
            # Use the most recent reset as the base
            original_capital = capital_resets[-1]['new_total']
        elif capital_additions:
            # Calculate original capital from first addition
            original_capital = capital_additions[0].get('total_capital_after', self.total_capital)
        
        return original_capital * 0.10  # 10% of original capital

# Example usage and testing
if __name__ == "__main__":
    # Test the portfolio manager
    portfolio = PortfolioManager()
    
    print("=== Portfolio Manager Test ===")
    print(f"Initial state: {portfolio.get_portfolio_summary()}")
    
    # Test opening a trade
    portfolio.open_trade("RELIANCE.NS", 1400.0, 1000.0, {"stop_loss": 1393.0, "target": 1450.0})
    print(f"After opening trade: {portfolio.get_portfolio_summary()}")
    
    # Test closing a trade with profit
    portfolio.close_trade("RELIANCE.NS", 1420.0, "Target reached")
    print(f"After closing trade: {portfolio.get_portfolio_summary()}")
    
    # Print performance stats
    print(f"Performance: {portfolio.get_performance_stats()}")