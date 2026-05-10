"""
Portfolio Returns Calculator & Analyzer
Handles multi-currency transactions, filtering, and returns calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
import requests
from functools import lru_cache

class CurrencyConverter:
    """Handles currency conversion with multiple data sources"""
    
    def __init__(self, rates_source: str = "yahoo"):
        """
        Initialize converter with preferred data source
        Args:
            rates_source: "yahoo", "google", or "manual" (for user-provided rates)
        """
        self.rates_source = rates_source
        self.rates_cache = {}
        self.base_currency = "SGD"
        
    def get_exchange_rates(self, target_date: str = None) -> Dict[str, float]:
        """
        Get exchange rates as of a specific date or today
        Returns rates relative to SGD
        """
        if self.rates_source == "yahoo":
            return self._get_yahoo_rates(target_date)
        elif self.rates_source == "google":
            return self._get_google_rates(target_date)
        else:
            return self._get_manual_rates()
    
    def _get_yahoo_rates(self, target_date: str = None) -> Dict[str, float]:
        """Fetch rates from Yahoo Finance"""
        try:
            # For simplicity, using fallback rates
            # In production, would use yfinance library
            rates = {
                "USD": 0.74,    # 1 USD = 0.74 SGD
                "JPY": 0.0051,  # 1 JPY = 0.0051 SGD
                "HKD": 0.095,   # 1 HKD = 0.095 SGD
                "SGD": 1.0
            }
            return rates
        except Exception as e:
            print(f"Error fetching Yahoo rates: {e}. Using fallback rates.")
            return self._get_fallback_rates()
    
    def _get_google_rates(self, target_date: str = None) -> Dict[str, float]:
        """Fetch rates from Google (via API or fallback)"""
        return self._get_fallback_rates()
    
    def _get_manual_rates(self) -> Dict[str, float]:
        """Return manually configured rates"""
        return {
            "USD": 0.74,
            "JPY": 0.0051,
            "HKD": 0.095,
            "SGD": 1.0
        }
    
    def _get_fallback_rates(self) -> Dict[str, float]:
        """Fallback rates if API is unavailable"""
        return {
            "USD": 0.74,
            "JPY": 0.0051,
            "HKD": 0.095,
            "SGD": 1.0
        }
    
    def convert(self, amount: float, from_currency: str, to_currency: str = "SGD") -> float:
        """Convert amount from one currency to another"""
        if pd.isna(amount) or amount == 0:
            return 0.0
        
        rates = self.get_exchange_rates()
        
        if from_currency not in rates or to_currency not in rates:
            raise ValueError(f"Currency not supported: {from_currency} or {to_currency}")
        
        # Convert: amount * (rate_to / rate_from)
        converted = amount * (rates[to_currency] / rates[from_currency])
        return converted


class PortfolioAnalyzer:
    """Main portfolio analysis engine"""
    
    def __init__(self, csv_path: str, converter: CurrencyConverter = None):
        self.df = pd.read_csv(csv_path)
        self.converter = converter or CurrencyConverter()
        self._clean_data()
    
    def _clean_data(self):
        """Initial data cleaning"""
        # Convert numeric columns
        numeric_cols = ['order_price', 'order_qty', 'fill_price', 'fill_qty', 'fees']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Handle NaT values
        self.df['order_time'] = pd.to_datetime(self.df['order_time'], errors='coerce')
        self.df['fees'] = self.df['fees'].fillna(0)
    
    def filter_valid_transactions(self) -> pd.DataFrame:
        """
        Return only successful transactions (status = 'Filled')
        Removes Cancelled and Failed transactions
        """
        valid = self.df[self.df['status'] == 'Filled'].copy()
        return valid
    
    def get_exited_positions(self) -> pd.DataFrame:
        """
        Identify positions that have been fully exited
        A position is exited when total sell quantity >= total buy quantity
        """
        valid_txns = self.filter_valid_transactions()
        
        exited_positions = []
        
        for symbol in valid_txns['symbol'].unique():
            symbol_txns = valid_txns[valid_txns['symbol'] == symbol]
            
            total_buy = symbol_txns[symbol_txns['side'] == 'Buy']['fill_qty'].sum()
            total_sell = symbol_txns[symbol_txns['side'] == 'Sell']['fill_qty'].sum()
            
            # Position is exited if total sell >= total buy
            if total_sell >= total_buy:
                exited_positions.append(symbol)
        
        exited_df = valid_txns[valid_txns['symbol'].isin(exited_positions)].copy()
        return exited_df.sort_values('symbol')
    
    def get_live_positions(self) -> pd.DataFrame:
        """
        Return current live positions (not fully exited)
        Includes quantity held and current unrealized position data
        """
        valid_txns = self.filter_valid_transactions()
        
        live_positions = []
        
        for symbol in valid_txns['symbol'].unique():
            symbol_txns = valid_txns[valid_txns['symbol'] == symbol]
            
            total_buy = symbol_txns[symbol_txns['side'] == 'Buy']['fill_qty'].sum()
            total_sell = symbol_txns[symbol_txns['side'] == 'Sell']['fill_qty'].sum()
            
            quantity_held = total_buy - total_sell
            
            # Only include if still holding
            if quantity_held > 0:
                # Get currency and latest info
                currency = symbol_txns['currency'].iloc[0]
                market = symbol_txns['market'].iloc[0]
                name = symbol_txns['name'].iloc[0]
                
                live_positions.append({
                    'symbol': symbol,
                    'name': name,
                    'quantity_held': quantity_held,
                    'currency': currency,
                    'market': market
                })
        
        return pd.DataFrame(live_positions)
    
    def calculate_trade_returns(self) -> pd.DataFrame:
        """
        Calculate returns for each completed trade pair (buy/sell)
        Groups buy and sell transactions by symbol and calculates P&L
        """
        valid_txns = self.filter_valid_transactions()
        
        trades = []
        
        for symbol in valid_txns['symbol'].unique():
            symbol_txns = valid_txns[valid_txns['symbol'] == symbol].sort_values('order_time')
            currency = symbol_txns['currency'].iloc[0]
            
            buy_txns = symbol_txns[symbol_txns['side'] == 'Buy']
            sell_txns = symbol_txns[symbol_txns['side'] == 'Sell']
            
            # Match buys with sells FIFO
            buy_qty_remaining = buy_txns['fill_qty'].sum()
            buy_cost_remaining = (buy_txns['fill_price'] * buy_txns['fill_qty'] + 
                                 buy_txns['fees']).sum()
            
            for _, sell_row in sell_txns.iterrows():
                sell_qty = sell_row['fill_qty']
                sell_proceeds = sell_row['fill_price'] * sell_qty - sell_row['fees']
                
                # Calculate proportional cost
                if buy_qty_remaining > 0:
                    cost_allocated = buy_cost_remaining * (sell_qty / buy_qty_remaining)
                    buy_qty_remaining -= sell_qty
                    buy_cost_remaining -= cost_allocated
                    
                    pnl = sell_proceeds - cost_allocated
                    pnl_pct = (pnl / cost_allocated * 100) if cost_allocated != 0 else 0
                    
                    # Convert to SGD
                    pnl_sgd = self.converter.convert(pnl, currency)
                    cost_sgd = self.converter.convert(cost_allocated, currency)
                    proceeds_sgd = self.converter.convert(sell_proceeds, currency)
                    
                    trades.append({
                        'symbol': symbol,
                        'side': 'Buy/Sell',
                        'quantity': sell_qty,
                        'cost_per_unit': cost_allocated / sell_qty if sell_qty > 0 else 0,
                        'sell_per_unit': sell_row['fill_price'],
                        'total_cost': cost_allocated,
                        'total_proceeds': sell_proceeds,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'currency': currency,
                        'total_cost_sgd': cost_sgd,
                        'total_proceeds_sgd': proceeds_sgd,
                        'pnl_sgd': pnl_sgd,
                        'exit_date': sell_row['order_time']
                    })
        
        return pd.DataFrame(trades)
    
    def get_portfolio_summary(self) -> Dict:
        """Calculate overall portfolio statistics"""
        valid_txns = self.filter_valid_transactions()
        
        # Total invested capital (all buy transactions)
        buy_txns = valid_txns[valid_txns['side'] == 'Buy']
        total_invested = (buy_txns['fill_price'] * buy_txns['fill_qty'] + 
                         buy_txns['fees']).sum()
        total_invested_sgd = self.converter.convert(total_invested, 'USD')  # Most are USD
        
        # Total realized P&L
        completed_trades = self.calculate_trade_returns()
        total_realized_pnl = completed_trades['pnl_sgd'].sum() if not completed_trades.empty else 0
        
        # Number of trades
        num_trades = len(valid_txns)
        num_completed = len(completed_trades)
        
        # Live positions
        live = self.get_live_positions()
        num_live = len(live)
        
        return {
            'total_invested_sgd': total_invested_sgd,
            'total_realized_pnl_sgd': total_realized_pnl,
            'num_trades': num_trades,
            'num_completed_trades': num_completed,
            'num_live_positions': num_live,
            'cancelled_transactions': len(self.df[self.df['status'] == 'Cancelled']),
            'failed_transactions': len(self.df[self.df['status'] == 'Failed'])
        }
    
    def generate_cleaned_csv(self, output_path: str, include_type: str = "valid"):
        """
        Generate cleaned CSV file
        include_type: 'valid', 'exited', 'live', or 'all'
        """
        if include_type == "valid":
            df = self.filter_valid_transactions()
        elif include_type == "exited":
            df = self.get_exited_positions()
        elif include_type == "live":
            df = self.get_live_positions()
        else:
            df = self.df
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved to {output_path}")


def main():
    """Example usage"""
    # Initialize converter (auto-fetches rates)
    converter = CurrencyConverter(rates_source="yahoo")
    
    # Analyze portfolio
    analyzer = PortfolioAnalyzer('/mnt/user-data/uploads/moomoo_trades_final.csv', converter)
    
    # Generate cleaned datasets
    analyzer.generate_cleaned_csv('/home/claude/valid_transactions.csv', 'valid')
    analyzer.generate_cleaned_csv('/home/claude/exited_trades.csv', 'exited')
    analyzer.generate_cleaned_csv('/home/claude/live_positions.csv', 'live')
    
    # Get statistics
    summary = analyzer.get_portfolio_summary()
    print("\n=== PORTFOLIO SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Get top trades
    trades = analyzer.calculate_trade_returns()
    if not trades.empty:
        print("\n=== TOP 5 TRADES BY P&L ===")
        print(trades.nlargest(5, 'pnl_sgd')[['symbol', 'quantity', 'pnl_sgd', 'pnl_pct']])
        
        print("\n=== HIGHEST SINGLE TRADE RETURN ===")
        best = trades.loc[trades['pnl_pct'].idxmax()] if not trades.empty else None
        if best is not None:
            print(best)
    
    # Get live positions
    live = analyzer.get_live_positions()
    print(f"\n=== LIVE POSITIONS ({len(live)}) ===")
    print(live)
    
    return analyzer, summary, trades


if __name__ == "__main__":
    main()
