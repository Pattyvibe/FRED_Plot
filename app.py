import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import warnings

warnings.filterwarnings('ignore')  # Suppress interpolation warnings

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
This app visualizes key US economic indicators with recession shading. 
Use it to gauge market sentiment: Negative yield spreads often precede recessions (fear signals), 
while high debt or inflation growth might indicate overleveraging (greed warnings).
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
            yield_spread = yield_spread.resample('MS').mean().reindex(monthly_index)  # No ffill; let trail NaN
            nasdaq = nasdaq.resample('MS').mean().reindex(monthly_index)  # No ffill
            jobless_claims = jobless_claims.resample('MS').mean().reindex(monthly_index)  # No ffill
            real_gdp = real_gdp.reindex(monthly_index).interpolate(method='linear')  # Interpolates between points only
            recession = recession.reindex(monthly_index).ffill()  # Keep ffill for indicator persistence
            nasdaq['Nasdaq_YoY'] = nasdaq['Nasdaq'].pct_change(12) * 100
            real_gdp['Real_GDP_YoY'] = real_gdp['Real_GDP'].pct_change(12) * 100
            data1 = pd.concat([yield_spread, nasdaq['Nasdaq_YoY'], jobless_claims, real_gdp['Real_GDP_YoY'], recession], axis=1)  # No dropna; allow uneven

            # Create first figure
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

            st.subheader("Latest Values (First Set)")
            for col in ['Yield_Spread', 'Nasdaq_YoY', 'Jobless_Claims', 'Real_GDP_YoY']:
                latest_idx = data1[col].last_valid_index()
                if latest_idx is not None:
                    latest_date = latest_idx.strftime('%Y-%m-%d')
                    latest_val = data1[col].loc[latest_idx]
                    if col == 'Yield_Spread':
                        st.write(f"10y-2y Yield Spread: {latest_val:.2f}% (as of {latest_date})")
                    elif col == 'Nasdaq_YoY':
                        st.write(f"NASDAQ Composite YoY Change: {latest_val:.2f}% (as of {latest_date})")
                    elif col == 'Jobless_Claims':
                        st.write(f"Initial Jobless Claims: {latest_val:.0f} thousand (as of {latest_date})")
                    elif col == 'Real_GDP_YoY':
                        st.write(f"Real GDP YoY: {latest_val:.2f}% (as of {latest_date})")

        # --- Second Plot: Inflation with Debt YoY ---
        m2 = fetch_series('M2SL', 'M2')
        gdp = fetch_series('GDPC1', 'Real_GDP')
        velocity = fetch_series('M2V', 'Velocity')
        cpi = fetch_series('CPIAUCSL', 'CPI')
        debt = fetch_series('GFDEBTN', 'Debt')  # Total public debt
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
            m2 = m2.reindex(monthly_index)  # No ffill (M2SL is monthly, so aligned)
            cpi = cpi.reindex(monthly_index)  # No ffill
            debt = debt.reindex(monthly_index).interpolate(method='linear')  # Interpolates between quarterly points
            recession = recession.reindex(monthly_index).ffill()  # Keep for indicator
            data2 = pd.concat([m2, gdp, velocity, cpi, debt, recession], axis=1)  # No dropna

            # Calculate growth rates (NaNs propagate naturally)
            data2['M2_Growth'] = data2['M2'].pct_change(12) * 100
            data2['GDP_Growth'] = data2['Real_GDP'].pct_change(12) * 100
            data2['Velocity_Growth'] = data2['Velocity'].pct_change(12) * 100
            data2['CPI_Inflation'] = data2['CPI'].pct_change(12) * 100
            data2['Debt_YoY'] = data2['Debt'].pct_change(12) * 100
            data2['Real_Inflation'] = data2['M2_Growth'] + data2['Velocity_Growth'] - data2['GDP_Growth']

            # Create second figure
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

            st.subheader("Latest Estimates (Second Set)")
            for col in ['Real_Inflation', 'CPI_Inflation', 'Debt_YoY']:
                latest_idx = data2[col].last_valid_index()
                if latest_idx is not None:
                    latest_date = latest_idx.strftime('%Y-%m-%d')
                    latest_val = data2[col].loc[latest_idx]
                    if col == 'Real_Inflation':
                        st.write(f"M2-Based Inflation: {latest_val:.2f}% (as of {latest_date})")
                    elif col == 'CPI_Inflation':
                        st.write(f"CPI Inflation: {latest_val:.2f}% (as of {latest_date})")
                    elif col == 'Debt_YoY':
                        st.write(f"US Debt YoY % Change: {latest_val:.2f}% (as of {latest_date})")

    st.success("Data refreshed successfully!")
