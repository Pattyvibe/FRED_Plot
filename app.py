import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
import datetime
import warnings
try:
    import requests
except ImportError:
    from urllib.request import urlopen
    import json
warnings.filterwarnings('ignore') # Suppress interpolation warnings
# Try to import plotly; if not installed, fall back to matplotlib
try:
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    has_plotly = True
except ImportError:
    has_plotly = False
    st.warning("Plotly not installed. Falling back to non-interactive matplotlib plots. Install plotly with 'pip install plotly' for interactive features.")
# Initialize FRED API (use secrets in production: st.secrets["FRED_API_KEY"])
fred = Fred(api_key='1f748bff3baca7c52ef515c02feada18')
# Function to fetch FRED series
def fetch_series(series_id, name):
    try:
        data = fred.get_series(series_id)
        if data.empty:
            raise ValueError(f"No data returned for {series_id}")
        return pd.DataFrame(data, columns=[name])
    except Exception as e:
        st.error(f"Error fetching {series_id}: {e}")
        return None
st.title("Economic Indicators Dashboard")
st.markdown("""
This app visualizes key US economic indicators with recession shading, inflation/debt trends, and BTC vs. Gold with Fear & Greed Index.
Use it to gauge market sentiment: Negative yield spreads often precede recessions (fear signals for Gold buys),
while high debt or inflation growth might indicate overleveraging (greed signals for BTC rallies).
""")
if st.button("Refresh Data"):
    with st.spinner("Fetching and processing data..."):
        # --- First Plot: Economic Indicators ---
        yield_spread = fetch_series('T10Y2Y', 'Yield_Spread')
        nasdaq = fetch_series('NASDAQCOM', 'Nasdaq')
        jobless_claims = fetch_series('ICSA', 'Jobless_Claims')
        real_gdp = fetch_series('GDPC1', 'Real_GDP')
        recession = fetch_series('USREC', 'Recession')
        if any(x is None for x in [yield_spread, nasdaq, jobless_claims, real_gdp, recession]):
            st.error("Failed to fetch one or more series for the first plot.")
        else:
            # Align data to monthly, allowing uneven ends
            start_date = max(pd.to_datetime('1992-01-01'), yield_spread.index.min(), nasdaq.index.min(), jobless_claims.index.min(), real_gdp.index.min(), recession.index.min())
            end_date = max(yield_spread.index.max(), nasdaq.index.max(), jobless_claims.index.max(), real_gdp.index.max(), recession.index.max())
            monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')
            yield_spread = yield_spread.resample('MS').mean().reindex(monthly_index) # No ffill; let trail NaN
            nasdaq = nasdaq.resample('MS').mean().reindex(monthly_index) # No ffill
            jobless_claims = jobless_claims.resample('MS').mean().reindex(monthly_index) # No ffill
            real_gdp = real_gdp.reindex(monthly_index).interpolate(method='linear') # Interpolates between points only
            recession = recession.reindex(monthly_index).ffill() # Keep ffill for indicator persistence
            nasdaq['Nasdaq_YoY'] = nasdaq['Nasdaq'].pct_change(12) * 100
            real_gdp['Real_GDP_YoY'] = real_gdp['Real_GDP'].pct_change(12) * 100
            data1 = pd.concat([yield_spread, nasdaq['Nasdaq_YoY'], jobless_claims, real_gdp['Real_GDP_YoY'], recession], axis=1) # No dropna; allow uneven
            st.subheader("Economic Indicators Explanation")
            st.markdown("""
            - **10Y-2Y Yield Spread**: A narrowing or inverted spread often signals economic slowdowns or recessions, prompting investors to shift from stocks/crypto to safer assets like bonds or gold.
            - **NASDAQ YoY % Change**: Reflects tech/growth stock performance; positive trends boost crypto sentiment, while declines signal risk-off moves.
            - **Initial Jobless Claims**: Rising claims indicate labor market weakness, often leading to stock/crypto sell-offs; low claims support bull markets.
            - **Real GDP YoY % Change**: Strong growth fuels stock/crypto rallies; slowing or negative growth raises recession fears, increasing volatility.
            """)
            if has_plotly:
                fig1_int = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                fig1_int.add_trace(go.Scatter(x=data1.index, y=data1['Yield_Spread'], mode='lines', name='Yield Spread', line=dict(color='blue')), row=1, col=1)
                fig1_int.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5, row=1, col=1)
                fig1_int.add_trace(go.Scatter(x=data1.index, y=data1['Nasdaq_YoY'], mode='lines', name='Nasdaq YoY', line=dict(color='green')), row=2, col=1)
                fig1_int.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5, row=2, col=1)
                fig1_int.add_trace(go.Scatter(x=data1.index, y=data1['Jobless_Claims'], mode='lines', name='Jobless Claims', line=dict(color='orange')), row=3, col=1)
                fig1_int.add_trace(go.Scatter(x=data1.index, y=data1['Real_GDP_YoY'], mode='lines', name='Real GDP YoY', line=dict(color='purple')), row=4, col=1)
                fig1_int.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5, row=4, col=1)
                # Add recession shades
                for row in range(1, 5):
                    i = 0
                    while i < len(data1['Recession']):
                        if data1['Recession'].iloc[i] == 1:
                            start_date = data1.index[i]
                            while i < len(data1['Recession']) and data1['Recession'].iloc[i] == 1:
                                i += 1
                            end_date = data1.index[i-1] if i > 0 else start_date
                            fig1_int.add_vrect(x0=start_date, x1=end_date, fillcolor="lightgrey", opacity=0.7, line_width=0, row=row, col=1)
                        else:
                            i += 1
                fig1_int.update_layout(height=800, title_text="Economic Indicators")
                st.subheader("Economic Indicators Plot (Interactive)")
                st.plotly_chart(fig1_int, use_container_width=True)
            else:
                # Fallback to matplotlib
                fig1 = plt.figure(figsize=(12, 12))
                axes = fig1.subplots(4, 1, sharex=True)
                axes[0].plot(data1.index, data1['Yield_Spread'], color='blue')
                axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[0].set_title('10y-2y Treasury Yield Spread')
                axes[0].set_ylabel('Spread (%)')
                axes[0].grid(True)
                axes[1].plot(data1.index, data1['Nasdaq_YoY'], color='green')
                axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[1].set_title('NASDAQ Composite YoY % Change')
                axes[1].set_ylabel('% Change')
                axes[1].grid(True)
                axes[2].plot(data1.index, data1['Jobless_Claims'], color='orange')
                axes[2].set_title('Initial Jobless Claims')
                axes[2].set_ylabel('Claims (Thousands)')
                axes[2].set_ylim(top=750000)
                axes[2].grid(True)
                axes[3].plot(data1.index, data1['Real_GDP_YoY'], color='purple')
                axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[3].set_title('Real GDP YoY % Change')
                axes[3].set_ylabel('% Change')
                axes[3].set_xlabel('Date')
                axes[3].grid(True)
                for ax in axes:
                    recession_periods = data1['Recession']
                    i = 0
                    while i < len(recession_periods):
                        if recession_periods.iloc[i] == 1:
                            start_date = data1.index[i]
                            while i < len(recession_periods) and recession_periods.iloc[i] == 1:
                                i += 1
                            end_date = data1.index[i-1] if i > 0 else start_date
                            ax.axvspan(start_date, end_date, color='lightgrey', alpha=0.7)
                        else:
                            i += 1
                fig1.tight_layout()
                st.subheader("Economic Indicators Plot")
                st.pyplot(fig1)
        # --- Second Plot: Inflation with Debt YoY ---
        m2 = fetch_series('M2SL', 'M2')
        gdp = fetch_series('GDPC1', 'Real_GDP')
        velocity = fetch_series('M2V', 'Velocity')
        cpi = fetch_series('CPIAUCSL', 'CPI')
        debt = fetch_series('GFDEBTN', 'Debt') # Total public debt
        recession = fetch_series('USREC', 'Recession')
        if any(x is None for x in [m2, gdp, velocity, cpi, debt, recession]):
            st.error("Failed to fetch one or more series for the second plot.")
        else:
            # Align data to monthly, allowing uneven ends
            start_date = max(m2.index.min(), gdp.index.min(), velocity.index.min(), cpi.index.min(), debt.index.min(), recession.index.min())
            end_date = max(m2.index.max(), gdp.index.max(), velocity.index.max(), cpi.index.max(), debt.index.max(), recession.index.max())
            monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')
            gdp = gdp.reindex(monthly_index).interpolate(method='linear')
            velocity = velocity.reindex(monthly_index).interpolate(method='linear')
            m2 = m2.reindex(monthly_index) # No ffill (M2SL is monthly, so aligned)
            cpi = cpi.reindex(monthly_index) # No ffill
            debt = debt.reindex(monthly_index).interpolate(method='linear') # Interpolates between quarterly points
            recession = recession.reindex(monthly_index).ffill() # Keep for indicator
            data2 = pd.concat([m2, gdp, velocity, cpi, debt, recession], axis=1) # No dropna
            # Calculate growth rates (NaNs propagate naturally)
            data2['M2_Growth'] = data2['M2'].pct_change(12) * 100
            data2['GDP_Growth'] = data2['Real_GDP'].pct_change(12) * 100
            data2['Velocity_Growth'] = data2['Velocity'].pct_change(12) * 100
            data2['CPI_Inflation'] = data2['CPI'].pct_change(12) * 100
            data2['Debt_YoY'] = data2['Debt'].pct_change(12) * 100
            data2['Real_Inflation'] = data2['M2_Growth'] + data2['Velocity_Growth'] - data2['GDP_Growth']
            st.subheader("Inflation and Debt Explanation")
            st.markdown("""
            - **M2-Based Inflation**: Based on Milton Friedman's monetarist theory that "inflation is always and everywhere a monetary phenomenon," this measures inflationary pressure from money supply growth (M2) adjusted for velocity and GDP. Unlike traditional demand-pull or cost-push explanations, it emphasizes excessive money creation as the root cause. High readings can erode stock/crypto purchasing power, favoring hard assets like BTC/gold.
            - **CPI Inflation**: Tracks consumer price changes; persistent highs signal Fed tightening, hurting growth-sensitive stocks/crypto, while moderation supports rallies.
            - **Debt YoY % Change**: Rapid debt growth can fuel short-term booms but risks long-term instability, devaluing fiat and boosting crypto as a hedge.
            """)
            if has_plotly:
                fig2_int = go.Figure()
                fig2_int.add_trace(go.Scatter(x=data2.index, y=data2['Real_Inflation'], mode='lines', name='M2-Based Inflation', line=dict(color='blue')))
                fig2_int.add_trace(go.Scatter(x=data2.index, y=data2['CPI_Inflation'], mode='lines', name='CPI Inflation', line=dict(color='red', dash='dash')))
                fig2_int.add_trace(go.Scatter(x=data2.index, y=data2['Debt_YoY'], mode='lines', name='Debt YoY % Change', line=dict(color='green', dash='dot')))
                recession_periods = data2['Recession']
                i = 0
                while i < len(recession_periods):
                    if recession_periods.iloc[i] == 1:
                        start_date = data2.index[i]
                        while i < len(recession_periods) and recession_periods.iloc[i] == 1:
                            i += 1
                        end_date = data2.index[i-1] if i > 0 else start_date
                        fig2_int.add_vrect(x0=start_date, x1=end_date, fillcolor="lightgrey", opacity=0.25, line_width=0)
                    else:
                        i += 1
                fig2_int.update_layout(title='M2-Based Inflation, CPI Inflation, and US Debt YoY % Change with US Recessions', xaxis_title='Date', yaxis_title='Rate (%)')
                st.subheader("Inflation and Debt Plot (Interactive)")
                st.plotly_chart(fig2_int, use_container_width=True)
            else:
                # Fallback to matplotlib
                fig2 = plt.figure(figsize=(12, 6))
                plt.plot(data2.index, data2['Real_Inflation'], label='M2-Based Inflation', color='blue')
                plt.plot(data2.index, data2['CPI_Inflation'], label='CPI Inflation', color='red', linestyle='--')
                plt.plot(data2.index, data2['Debt_YoY'], label='Debt YoY % Change', color='green', linestyle='-.')
                recession_periods = data2['Recession']
                i = 0
                while i < len(recession_periods):
                    if recession_periods.iloc[i] == 1:
                        start_date = data2.index[i]
                        while i < len(recession_periods) and recession_periods.iloc[i] == 1:
                            i += 1
                        end_date = data2.index[i-1] if i > 0 else start_date
                        plt.axvspan(start_date, end_date, color='lightgrey', alpha=0.25)
                    else:
                        i += 1
                plt.title('M2-Based Inflation, CPI Inflation, and US Debt YoY % Change with US Recessions')
                plt.xlabel('Date')
                plt.ylabel('Rate (%)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                st.subheader("Inflation and Debt Plot")
                st.pyplot(fig2)
            st.subheader("Explanation of M2-Based Inflation")
            st.markdown("""
            The M2-Based Inflation is derived from the Quantity Theory of Money, which posits that MV = PY, where:
            - M is the money supply (here, M2),
            - V is the velocity of money,
            - P is the price level (inflation when considering growth),
            - Y is real output (GDP).
            
            Rearranging for the growth rates, the approximate inflation rate (ΔP) is calculated as:
            
            ΔM (M2 Growth) + ΔV (Velocity Growth) - ΔY (GDP Growth),
            
            where all terms are year-over-year percentage changes.
            
            This provides an alternative measure of inflationary pressure based on monetary factors.
            """)
        # --- Third Plot: BTC vs Gold with Separate Fear & Greed Plot ---
        start = '2021-01-01'
        end = datetime.date.today().strftime('%Y-%m-%d')

        # Fetch price data
        try:
            btc = yf.download('BTC-USD', start=start, end=end, auto_adjust=True)
            gold = yf.download('GC=F', start=start, end=end, auto_adjust=True)
        except Exception as e:
            st.error(f"Error fetching price data: {str(e)}")
            btc, gold = pd.DataFrame(), pd.DataFrame()

        # Handle multi-index columns
        if isinstance(btc.columns, pd.MultiIndex):
            btc = btc.droplevel(1, axis=1)
        if isinstance(gold.columns, pd.MultiIndex):
            gold = gold.droplevel(1, axis=1)

        if btc.empty or gold.empty:
            st.error("Error: No price data fetched. Check internet, ticker, or date range.")
        else:
            # Combine prices
            df = pd.DataFrame({
                'BTC': btc['Close'],
                'Gold': gold['Close']
            }).dropna()

            # Fetch Fear & Greed data
            url = 'https://api.alternative.me/fng/?limit=0'
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                st.error(f"Error fetching F&G: {str(e)}")
                try:
                    with urlopen(url) as response:
                        data = json.loads(response.read().decode())
                except Exception as e2:
                    st.error(f"Fallback F&G fetch failed: {str(e2)}")
                    data = {'data': []}

            if data.get('data'):
                fg_df = pd.DataFrame(data['data'])
                # Keep relevant columns
                fg_df = fg_df[['value', 'timestamp']]
                fg_df['timestamp'] = pd.to_numeric(fg_df['timestamp'])  # String to numeric
                fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], unit='s')
                fg_df['value'] = pd.to_numeric(fg_df['value'])
                fg_df = fg_df.set_index('timestamp')[['value']]
                # Align to df's index (nearest for gaps, e.g., weekends)
                fg_df = fg_df.reindex(df.index, method='nearest')
            else:
                st.error("No F&G data available.")
                fg_df = pd.DataFrame()

            if df.empty:
                st.error("Error: No overlapping data after alignment. Try different dates.")
            else:
                st.subheader("BTC vs Gold Explanation")
                st.markdown("""
                Comparing BTC and gold prices highlights their roles as inflation hedges and safe havens; BTC often amplifies gold trends in risk-on environments, while divergences signal shifts in investor preference between 'digital gold' and traditional stores of value.
                """)
                # First Plot: BTC vs Gold
                if has_plotly:
                    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig1.add_trace(go.Scatter(x=df.index, y=df['BTC'], mode='lines', name='BTC Price (USD)', line=dict(color='darkorange')), secondary_y=False)
                    fig1.add_trace(go.Scatter(x=df.index, y=df['Gold'], mode='lines', name='Gold Futures (USD/oz)', line=dict(color='darkblue')), secondary_y=True)
                    fig1.update_layout(title='BTC Price (USD) vs. Gold Futures (USD/oz), 2021-Present', height=500)
                    fig1.update_yaxes(title_text='BTC Price (USD)', secondary_y=False, title_font=dict(color='darkorange'), tickfont=dict(color='darkorange'))
                    fig1.update_yaxes(title_text='Gold Price (USD/oz)', secondary_y=True, title_font=dict(color='darkblue'), tickfont=dict(color='darkblue'))
                    st.subheader("BTC vs Gold Plot (Interactive - Plotly)")
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    ax1.plot(df.index, df['BTC'], label='BTC Price (USD)', color='darkorange')
                    ax1.set_ylabel('BTC Price (USD)', color='darkorange')
                    ax1.tick_params(axis='y', labelcolor='darkorange')
                    ax2 = ax1.twinx()
                    ax2.plot(df.index, df['Gold'], label='Gold Futures (USD/oz)', color='darkblue')
                    ax2.set_ylabel('Gold Price (USD/oz)', color='darkblue')
                    ax2.tick_params(axis='y', labelcolor='darkblue')
                    ax1.set_title('BTC Price (USD) vs. Gold Futures (USD/oz), 2021-Present')
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    fig1.tight_layout()
                    st.subheader("BTC vs Gold Plot (Matplotlib)")
                    st.pyplot(fig1)

                # Second Plot: Fear & Greed
                if has_plotly:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=fg_df.index, y=fg_df['value'], mode='lines', name='Fear & Greed Index', line=dict(color='green')))
                    fig2.add_hline(y=50, line_dash="dot", line_color="black", annotation_text="Neutral (50)")
                    fig2.update_yaxes(range=[0, 100], title_text='Fear & Greed (0-100)')
                    fig2.update_layout(title='Fear & Greed Index, 2021-Present', height=300)
                    st.subheader("Fear & Greed Plot (Interactive - Plotly)")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    fig2, ax3 = plt.subplots(figsize=(12, 4))
                    ax3.plot(fg_df.index, fg_df['value'], label='Fear & Greed Index', color='green')
                    ax3.axhline(50, color='black', linestyle='--', label='Neutral (50)')
                    ax3.set_ylabel('Fear & Greed (0-100)')
                    ax3.set_ylim(0, 100)
                    ax3.set_title('Fear & Greed Index, 2021-Present')
                    ax3.legend(loc='upper left')
                    fig2.tight_layout()
                    st.subheader("Fear & Greed Plot (Matplotlib)")
                    st.pyplot(fig2)
                st.image("https://alternative.me/crypto/fear-and-greed-index.png", caption="Latest Crypto Fear & Greed Index")
                st.markdown("""
                The Crypto Fear & Greed Index is a daily metric that analyzes emotions and sentiments from sources like volatility (25%), market momentum/volume (25%), social media (15%), Bitcoin dominance (10%), and trends (10%) to produce a value from 0 (Extreme Fear) to 100 (Extreme Greed). It helps identify market sentiment for better decision-making. Data sourced from [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/).
                """)
    st.success("Data refreshed successfully!")
