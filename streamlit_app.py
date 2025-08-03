import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from trading_engine import TradingEngine
from portfolio import PortfolioManager

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .stMetric > div > div > div > div {
        font-size: 0.8rem;
    }
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.3rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = TradingEngine()
    
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = PortfolioManager()

# Sidebar navigation
st.sidebar.title("Trading Dashboard")
page = st.sidebar.selectbox("Select Page", ["Trading Analysis", "Portfolio Manager"])

if page == "Trading Analysis":
    # ========== TRADING ANALYSIS PAGE ==========
    
    st.markdown('<div class="main-header">Trading Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Symbol management in sidebar
    st.sidebar.markdown('<div class="section-header">Manage Symbols</div>', unsafe_allow_html=True)
    
    # Add new symbol section
    st.sidebar.markdown("**Add New Symbol**")
    col_input, col_add_btn = st.sidebar.columns([2, 1])
    with col_input:
        new_symbol = st.text_input("Symbol", placeholder="e.g., HDFCBANK.NS", label_visibility="collapsed")
    with col_add_btn:
        add_button = st.button("➕ ADD", type="primary")
        if add_button and new_symbol:
            if st.session_state.engine.add_symbol(new_symbol.upper()):
                st.sidebar.success(f"✅ Added {new_symbol.upper()}")
                # Force refresh of symbols list
                st.session_state.engine.symbols = st.session_state.engine.load_symbols()
                time.sleep(0.1)  # Small delay to ensure file is written
                st.rerun()
            else:
                st.sidebar.warning(f"⚠️ {new_symbol.upper()} already exists")
                st.rerun()
    
    # Current symbols list - Always refresh from file
    current_symbols = st.session_state.engine.load_symbols()
    st.session_state.engine.symbols = current_symbols  # Update session state
    
    if current_symbols:
        st.sidebar.markdown("**Current Symbols**")
        st.sidebar.markdown("*Click the ❌ button to remove a symbol*")
        
        # Create a nice table-like display
        for i, symbol in enumerate(current_symbols):
            col_symbol, col_remove = st.sidebar.columns([3, 1])
            with col_symbol:
                st.markdown(f"**{i+1}.** {symbol}")
            with col_remove:
                remove_button = st.button("❌", key=f"remove_{symbol}", help=f"Remove {symbol}")
                if remove_button:
                    if st.session_state.engine.remove_symbol(symbol):
                        st.sidebar.success(f"🗑️ Removed {symbol}")
                        # Force refresh of symbols list
                        st.session_state.engine.symbols = st.session_state.engine.load_symbols()
                        time.sleep(0.1)  # Small delay to ensure file is written
                        st.rerun()
        
        # Show total count
        st.sidebar.info(f"📊 Total symbols: {len(current_symbols)}")
        
        # Bulk actions
        if len(current_symbols) > 1:
            st.sidebar.markdown("**Bulk Actions**")
            clear_all_button = st.sidebar.button("🗑️ CLEAR ALL SYMBOLS", type="secondary")
            if clear_all_button:
                # Clear all symbols
                for symbol in current_symbols.copy():
                    st.session_state.engine.remove_symbol(symbol)
                # Force refresh of symbols list
                st.session_state.engine.symbols = st.session_state.engine.load_symbols()
                st.sidebar.success("🗑️ All symbols cleared!")
                time.sleep(0.1)  # Small delay to ensure file is written
                st.rerun()
    else:
        st.sidebar.markdown("**Current Symbols**")
        st.sidebar.info("📝 No symbols added yet. Add your first symbol above!")
    
    # Analysis controls
    st.sidebar.markdown('<div class="section-header">Analysis</div>', unsafe_allow_html=True)
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing symbols..."):
            # Get active trades and portfolio info
            active_trades = st.session_state.portfolio.get_active_trades()
            portfolio_summary = st.session_state.portfolio.get_portfolio_summary()
            
            results = st.session_state.engine.run_analysis(active_trades, portfolio_summary['total_capital'])
            st.session_state.last_results = results
            
            # Process any exit signals
            for symbol, result in results['symbol_results'].items():
                if result.get('should_exit') and symbol in active_trades:
                    exit_info = result['exit_info']
                    st.session_state.portfolio.close_trade(
                        symbol, 
                        exit_info['exit_price'], 
                        exit_info['exit_reason']
                    )
                    st.sidebar.success(f"Closed {symbol}: {exit_info['exit_reason']}")
                    
            # Process any new entry signals (check if we can take more trades)
            for symbol, result in results['symbol_results'].items():
                if result.get('has_signal'):
                    signal = result['signal_info']
                    
                    # Check if we can open new trade (max 10 concurrent)
                    if not st.session_state.portfolio.can_open_new_trade():
                        st.sidebar.warning(f"Cannot open {symbol}: Maximum 10 trades already active")
                        continue
                    
                    # Use the calculated position size from signal
                    if st.session_state.portfolio.is_sufficient_capital(signal['position_size']):
                        # Auto-execute the trade
                        if st.session_state.portfolio.open_trade(
                            symbol, 
                            signal['entry_price'], 
                            signal['position_size'], 
                            signal
                        ):
                            st.sidebar.success(f"Opened trade: {symbol} (₹{signal['position_size']:,.2f})")
                    else:
                        st.sidebar.warning(f"Insufficient capital for {symbol}")
                        
        st.sidebar.success("Analysis completed!")
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="section-header">Symbol List</div>', unsafe_allow_html=True)
        
        # Status overview
        portfolio_summary = st.session_state.portfolio.get_portfolio_summary()
        capital_info = st.session_state.portfolio.get_capital_utilization_info()
        
        st.metric("Active Trades", f"{portfolio_summary['active_trades_count']}/10")
        st.metric("Trade Slots Available", capital_info['remaining_trade_slots'])
        st.metric("Position Size/Trade", f"₹{capital_info['position_size_per_trade']:,.0f}")
        
        # Symbol selection
        if current_symbols:
            selected_symbol = st.radio("Select Symbol for Chart:", current_symbols, key="symbol_radio")
        else:
            selected_symbol = None
            st.warning("No symbols added yet")
        
        # Show recent signals
        if hasattr(st.session_state, 'last_results'):
            st.markdown('<div class="section-header">Recent Signals</div>', unsafe_allow_html=True)
            results = st.session_state.last_results
            active_trades = st.session_state.portfolio.get_active_trades()
            
            for symbol, result in results['symbol_results'].items():
                if result['has_signal']:
                    st.markdown(f'<div class="success-box"><strong>{symbol}</strong><br>Entry: ₹{result["signal_info"]["entry_price"]:.2f}<br>Support: ₹{result["signal_info"]["support_level"]:.2f}</div>', unsafe_allow_html=True)
                elif symbol in active_trades:
                    st.markdown(f'<div class="info-box"><strong>{symbol}</strong> - Active Trade</div>', unsafe_allow_html=True)
                elif result.get('pin_bar_detected'):
                    st.markdown(f'<div class="warning-box"><strong>{symbol}</strong> - Pin Bar Detected</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">Chart Analysis</div>', unsafe_allow_html=True)
        
        if selected_symbol:
            # Chart controls - NOW PROPERLY POSITIONED
            st.markdown("### Chart Settings")
            col_length, col_candle = st.columns(2)
            
            with col_length:
                length = st.selectbox(
                    "Length",
                    options=["1 Day", "1 Month"],
                    index=1  # Default to 1 Month
                )
            
            with col_candle:
                candle = st.selectbox(
                    "Candle",
                    options=["15 Minutes", "1 Hour", "Daily"],
                    index=1  # Default to 1 Hour
                )
            
            # Validation: Daily candles only work with 1 Month length
            if candle == "Daily" and length == "1 Day":
                st.warning("Daily candles require 1 Month length. Switching to 1 Hour candles.")
                candle = "1 Hour"
            
            # Map to API parameters
            length_map = {
                "1 Day": "1d",
                "1 Month": "1mo"
            }
            
            candle_map = {
                "15 Minutes": "15m",
                "1 Hour": "1h", 
                "Daily": "1d"
            }
            
            # Show current selection
            st.info(f"📊 Displaying: {length} with {candle} candles")
            
            try:
                # Fetch data
                chart_data = st.session_state.engine.get_symbol_data(
                    selected_symbol, 
                    period=length_map[length], 
                    interval=candle_map[candle]
                )
                
                if chart_data is not None and not chart_data.empty:
                    # Create main chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'{selected_symbol} - {length} {candle} Chart', 'Volume'),
                        row_heights=[0.8, 0.2]
                    )
                    
                    # Candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=chart_data.index,
                            open=chart_data['Open'],
                            high=chart_data['High'],
                            low=chart_data['Low'],
                            close=chart_data['Close'],
                            name='Price',
                            increasing_line_color='green',
                            decreasing_line_color='red'
                        ),
                        row=1, col=1
                    )
                    
                    # Add support/resistance levels
                    engine_status = st.session_state.engine.get_status()
                    if selected_symbol in engine_status['support_resistance_levels']:
                        levels = engine_status['support_resistance_levels'][selected_symbol]
                        
                        # Support levels (green dashed lines)
                        for i, support in enumerate(levels.get('support', [])[-3:]):
                            fig.add_hline(
                                y=support, 
                                line_dash="dash", 
                                line_color="green",
                                line_width=1,
                                annotation_text=f"Support: ₹{support:.2f}",
                                annotation_position="bottom right" if i % 2 == 0 else "top right",
                                row=1, col=1
                            )
                        
                        # Resistance levels (red dashed lines)
                        for i, resistance in enumerate(levels.get('resistance', [])[:3]):
                            fig.add_hline(
                                y=resistance, 
                                line_dash="dash", 
                                line_color="red",
                                line_width=1,
                                annotation_text=f"Resistance: ₹{resistance:.2f}",
                                annotation_position="top left" if i % 2 == 0 else "bottom left",
                                row=1, col=1
                            )
                    
                    # Volume chart
                    colors = ['green' if chart_data['Close'].iloc[i] >= chart_data['Open'].iloc[i] 
                             else 'red' for i in range(len(chart_data))]
                    
                    fig.add_trace(
                        go.Bar(
                            x=chart_data.index,
                            y=chart_data['Volume'],
                            name='Volume',
                            marker_color=colors,
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=700,
                        showlegend=False,
                        xaxis_rangeslider_visible=False,
                        title_x=0.5
                    )
                    
                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Symbol info below chart - adapt to timeframe
                    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                    
                    current_price = chart_data['Close'].iloc[-1]
                    prev_close = chart_data['Close'].iloc[-2] if len(chart_data) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    with col_info1:
                        st.metric("Current Price", f"₹{current_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                    
                    with col_info2:
                        if length == "1 Day":
                            # Show day's high/low
                            day_high = chart_data['High'].max()
                            day_low = chart_data['Low'].min()
                            st.metric("Day High", f"₹{day_high:.2f}")
                            st.metric("Day Low", f"₹{day_low:.2f}")
                        else:  # 1 Month
                            # Show month's high/low
                            month_high = chart_data['High'].max()
                            month_low = chart_data['Low'].min()
                            st.metric("Month High", f"₹{month_high:.2f}")
                            st.metric("Month Low", f"₹{month_low:.2f}")
                    
                    with col_info3:
                        engine_status = st.session_state.engine.get_status()
                        if selected_symbol in engine_status['support_resistance_levels']:
                            levels = engine_status['support_resistance_levels'][selected_symbol]
                            support_levels = levels.get('support', [])
                            if support_levels:
                                nearest_support = max([s for s in support_levels if s <= current_price], default=0)
                                if nearest_support:
                                    st.metric("Nearest Support", f"₹{nearest_support:.2f}")
                                else:
                                    st.metric("Nearest Support", "None")
                            else:
                                st.metric("Nearest Support", "None")
                    
                    with col_info4:
                        if selected_symbol in engine_status['support_resistance_levels']:
                            levels = engine_status['support_resistance_levels'][selected_symbol]
                            resistance_levels = levels.get('resistance', [])
                            if resistance_levels:
                                nearest_resistance = min([r for r in resistance_levels if r >= current_price], default=0)
                                if nearest_resistance:
                                    st.metric("Nearest Resistance", f"₹{nearest_resistance:.2f}")
                                else:
                                    st.metric("Nearest Resistance", "None")
                            else:
                                st.metric("Nearest Resistance", "None")
                    
                    # Chart info summary
                    st.info(f"📊 Showing {length} data with {candle} candles | Data points: {len(chart_data)}")
                    
                    # Signal status
                    if hasattr(st.session_state, 'last_results'):
                        symbol_result = st.session_state.last_results['symbol_results'].get(selected_symbol, {})
                        if symbol_result.get('pin_bar_detected'):
                            st.markdown('<div class="success-box">Pin Bar Detected on Latest Candle</div>', unsafe_allow_html=True)
                        
                        if symbol_result.get('has_signal'):
                            signal = symbol_result['signal_info']
                            st.markdown('<div class="success-box"><strong>ENTRY SIGNAL ACTIVE</strong></div>', unsafe_allow_html=True)
                            col_signal1, col_signal2 = st.columns(2)
                            with col_signal1:
                                st.write(f"**Entry Price:** ₹{signal['entry_price']:.2f}")
                                st.write(f"**Support Level:** ₹{signal['support_level']:.2f}")
                            with col_signal2:
                                st.write(f"**Target:** ₹{signal['target']:.2f}" if signal['target'] else "**Target:** Not Set")
                                st.write(f"**Stop Loss:** ₹{signal['stop_loss']:.2f}")
                
                else:
                    st.error(f"Could not fetch data for {selected_symbol}")
                    
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
        else:
            st.info("Select a symbol from the list to view its chart")

elif page == "Portfolio Manager":
    # ========== PORTFOLIO MANAGEMENT PAGE ==========
    
    st.markdown('<div class="main-header">Portfolio Management</div>', unsafe_allow_html=True)
    
    # Portfolio overview
    portfolio_summary = st.session_state.portfolio.get_portfolio_summary()
    capital_info = st.session_state.portfolio.get_capital_utilization_info()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Capital", f"₹{portfolio_summary['total_capital']:,.2f}")
    with col2:
        st.metric("Available Amount", f"₹{portfolio_summary['available_amount']:,.2f}")
    with col3:
        st.metric("Deployed Amount", f"₹{portfolio_summary['deployed_amount']:,.2f}")
    with col4:
        st.metric("Capital Utilization", f"{portfolio_summary['capital_utilization_pct']:.1f}%")
    
    # Trading capacity overview
    st.markdown('<div class="section-header">Trading Capacity</div>', unsafe_allow_html=True)
    
    col_cap1, col_cap2, col_cap3, col_cap4 = st.columns(4)
    
    with col_cap1:
        st.metric("Active Trades", f"{capital_info['current_active_trades']}/10")
    with col_cap2:
        st.metric("Available Slots", capital_info['remaining_trade_slots'])
    with col_cap3:
        st.metric("Position Size/Trade", f"₹{capital_info['position_size_per_trade']:,.0f}")
    with col_cap4:
        st.metric("Max Total Deployment", f"₹{capital_info['total_potential_deployment']:,.0f}")
    
    # Portfolio management controls
    st.markdown('<div class="section-header">Portfolio Controls</div>', unsafe_allow_html=True)
    
    portfolio = st.session_state.portfolio
    col_control1, col_control2 = st.columns(2)
    
    with col_control1:
        st.write("**Add/Remove Capital**")
        capital_change = st.number_input("Amount", value=0.0, step=1000.0, format="%.2f")
        col_add, col_remove = st.columns(2)
        
        with col_add:
            if st.button("Add Capital"):
                if capital_change > 0:
                    if portfolio.add_capital(capital_change):
                        st.success(f"Added ₹{capital_change:,.2f} to portfolio")
                        st.rerun()
        
        with col_remove:
            if st.button("Remove Capital"):
                if capital_change > 0:
                    if portfolio.remove_capital(capital_change):
                        st.success(f"Removed ₹{capital_change:,.2f} from portfolio")
                        st.rerun()
                    else:
                        st.error("Cannot remove more than available amount")
    
    with col_control2:
        st.write("**Reset Portfolio**")
        portfolio_summary = st.session_state.portfolio.get_portfolio_summary()
        new_total = st.number_input("Set Total Capital", value=float(portfolio_summary['total_capital']), step=1000.0, format="%.2f")
        if st.button("Reset Portfolio"):
            if portfolio.set_total_capital(new_total):
                st.success("Portfolio reset successfully")
                st.rerun()
    
    # Active trades section
    st.markdown('<div class="section-header">Active Trades</div>', unsafe_allow_html=True)
    
    active_trades = st.session_state.portfolio.get_active_trades()
    if active_trades:
        trades_data = []
        
        for symbol, trade in active_trades.items():
            # Get current price for P&L calculation
            try:
                current_data = st.session_state.engine.get_symbol_data(symbol, period='1d', interval='1h')
                if current_data is not None and not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    trade_pnl = st.session_state.portfolio.update_trade_with_current_price(symbol, current_price)
                    pnl_amount = trade_pnl['unrealized_pnl'] if trade_pnl else 0
                    pnl_pct = trade_pnl['unrealized_pnl_pct'] if trade_pnl else 0
                else:
                    current_price = trade['entry_price']
                    pnl_amount = 0
                    pnl_pct = 0
            except:
                current_price = trade['entry_price']
                pnl_amount = 0
                pnl_pct = 0
            
            trades_data.append({
                'Symbol': symbol,
                'Entry Price': f"₹{trade['entry_price']:.2f}",
                'Current Price': f"₹{current_price:.2f}",
                'Amount Deployed': f"₹{trade['amount_deployed']:,.2f}",
                'P&L Amount': f"₹{pnl_amount:.2f}",
                'P&L %': f"{pnl_pct:.2f}%",
                'Target': f"₹{trade['trade_details'].get('target', 0):.2f}" if trade['trade_details'].get('target') else "N/A",
                'Stop Loss': f"₹{trade['trade_details'].get('stop_loss', 0):.2f}",
                'Entry Time': datetime.fromisoformat(trade['entry_time']).strftime('%d/%m %H:%M')
            })
        
        if trades_data:
            df_trades = pd.DataFrame(trades_data)
            st.dataframe(df_trades, use_container_width=True)
            
            # Trade summary
            total_deployed = sum([trade['amount_deployed'] for trade in active_trades.values()])
            total_pnl = sum([float(row['P&L Amount'].replace('₹', '').replace(',', '')) for row in trades_data])
            st.info(f"**Total Deployed:** ₹{total_deployed:,.2f} | **Total Unrealized P&L:** ₹{total_pnl:.2f}")
    else:
        st.info("No active trades currently")
    
    # Portfolio allocation chart
    st.markdown('<div class="section-header">Portfolio Allocation</div>', unsafe_allow_html=True)
    
    portfolio_summary = st.session_state.portfolio.get_portfolio_summary()
    if portfolio_summary['total_capital'] > 0:
        # Create pie chart
        labels = ['Available', 'Deployed']
        values = [portfolio_summary['available_amount'], portfolio_summary['deployed_amount']]
        colors = ['#00CC96', '#FF6B6B']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>₹%{value:,.0f}<br>(%{percent})'
        )])
        
        fig.update_layout(
            title="Capital Allocation",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance statistics
    st.markdown('<div class="section-header">Performance Statistics</div>', unsafe_allow_html=True)
    performance = st.session_state.portfolio.get_performance_stats()
    
    if performance['total_trades'] > 0:
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        with col_perf1:
            st.metric("Total Trades", performance['total_trades'])
            st.metric("Win Rate", f"{performance['win_rate']:.1f}%")
        
        with col_perf2:
            st.metric("Winning Trades", performance['winning_trades'])
            st.metric("Losing Trades", performance['losing_trades'])
        
        with col_perf3:
            st.metric("Total P&L", f"₹{performance['total_pnl']:.2f}")
            st.metric("Avg P&L/Trade", f"₹{performance['avg_pnl_per_trade']:.2f}")
        
        with col_perf4:
            st.metric("Best Trade", f"₹{performance['best_trade']:.2f}")
            st.metric("Worst Trade", f"₹{performance['worst_trade']:.2f}")
    else:
        st.info("No closed trades yet to show performance statistics")

# Auto-refresh option (only for trading analysis page)
if page == "Trading Analysis":
    st.sidebar.markdown('<div class="section-header">Auto Refresh</div>', unsafe_allow_html=True)
    auto_refresh = st.sidebar.checkbox("Enable Auto Refresh")
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
        time.sleep(refresh_interval)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Strategy:** Pin bar at support level | **Position Size:** 10% of capital per trade | **Max Trades:** 10 concurrent | **Stop Loss:** 0.5%")