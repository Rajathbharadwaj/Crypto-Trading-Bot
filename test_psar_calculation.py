import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

# Extract the data from Untitled.ipynb (sample data shown in conversation)
data = [
    (1732597500, 2625.616, 2625.92, 2624.306, 2624.976, 467, 112, 0),
    (1732597800, 2625.002, 2625.439, 2621.567, 2621.959, 713, 112, 0),
    (1732598100, 2621.935, 2622.967, 2621.863, 2622.347, 519, 112, 0),
    (1732598400, 2622.339, 2624.392, 2621.686, 2624.291, 477, 112, 0),
    (1732598700, 2624.277, 2624.348, 2622.099, 2622.516, 537, 112, 0),
    (1732599000, 2622.474, 2622.502, 2617.856, 2619.431, 1316, 111, 0),
    (1732599300, 2619.459, 2622.081, 2619.41, 2621.956, 951, 112, 0),
    (1732599600, 2621.991, 2623.276, 2620.519, 2621.409, 842, 111, 0),
    (1732599900, 2621.4, 2621.553, 2619.835, 2621.259, 772, 112, 0),
    (1732600200, 2621.239, 2622.033, 2619.876, 2621.82, 842, 112, 0),
    (1732600500, 2621.863, 2625.366, 2621.863, 2624.332, 1032, 112, 0),
    (1732600800, 2624.375, 2624.972, 2622.857, 2624.505, 934, 111, 0),
    (1732601100, 2624.535, 2625.627, 2623.177, 2624.718, 883, 111, 0),
    (1732601400, 2624.719, 2626.038, 2624.346, 2624.736, 936, 112, 0),
    (1732601700, 2624.683, 2625.037, 2622.648, 2622.736, 627, 112, 0),
    (1732602000, 2622.67, 2623.653, 2621.092, 2622.781, 852, 111, 0),
    (1732602300, 2622.765, 2623.753, 2621.213, 2623.212, 789, 111, 0),
    (1732602600, 2623.252, 2623.808, 2622.473, 2622.742, 750, 112, 0),
    (1732602900, 2622.647, 2623.559, 2621.916, 2622.848, 764, 112, 0),
    (1732603200, 2622.875, 2623.316, 2622.035, 2622.216, 613, 112, 0),
    (1732603500, 2622.268, 2622.746, 2621.561, 2622.137, 648, 112, 0),
    (1732603800, 2622.183, 2623.438, 2621.669, 2622.999, 648, 111, 0),
    (1732604100, 2623.037, 2623.203, 2621.812, 2622.155, 630, 111, 0),
    (1732604400, 2622.112, 2622.173, 2620.343, 2620.955, 578, 111, 0),
    (1732604700, 2621.024, 2622.55, 2621.024, 2621.926, 563, 111, 0),
    (1732605000, 2621.965, 2622.883, 2621.55, 2622.265, 681, 112, 0),
    (1732605300, 2622.255, 2622.774, 2621.881, 2622.251, 556, 111, 0),
    (1732605600, 2622.261, 2622.261, 2619.148, 2619.191, 667, 111, 0),
    (1732605900, 2619.135, 2619.2, 2615.882, 2616.177, 937, 112, 0),
    (1732606200, 2616.135, 2616.135, 2613.48, 2614.16, 926, 111, 0),
    (1732606500, 2614.193, 2614.523, 2612.225, 2614.353, 889, 112, 0),
    (1732606800, 2614.314, 2615.698, 2612.442, 2612.543, 972, 112, 0),
    (1732607100, 2612.505, 2613.11, 2610.855, 2612.581, 901, 112, 0),
    (1732607400, 2612.574, 2613.57, 2611.894, 2612.465, 856, 112, 0),
    (1732607700, 2612.452, 2614.188, 2612.09, 2614.06, 742, 112, 0),
    (1732608000, 2614.094, 2615.325, 2612.643, 2615.252, 1541, 112, 0),
    (1732608300, 2615.213, 2616.901, 2614.063, 2615.062, 1279, 112, 0),
    (1732608600, 2615.056, 2617.606, 2615.037, 2617.467, 803, 112, 0),
    (1732608900, 2617.577, 2618.307, 2615.906, 2617.148, 858, 111, 0),
    (1732609200, 2617.248, 2617.248, 2614.467, 2615.187, 776, 112, 0)
]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'volume', 'unknown'])

# Display the first few rows
console.print("Original Data:")
console.print(df[['open', 'high', 'low', 'close']].head())

# Calculate PSAR using the exact same parameters as in our trading bots
psar_step = 0.02
psar_max = 0.2

# Method 1: Using pandas_ta - this is what our bot is using
psar_ta = ta.psar(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    af0=psar_step,
    af=psar_step,
    max_af=psar_max
)

# Print the PSAR column names to verify
psar_columns = list(psar_ta.columns)
console.print(f"PSAR columns from pandas_ta: {psar_columns}")

# Get the long and short PSAR columns
psar_l_col = f'PSARl_{psar_step}_{psar_max}'
psar_s_col = f'PSARs_{psar_step}_{psar_max}'

# Check if expected columns exist
if psar_l_col not in psar_ta.columns or psar_s_col not in psar_ta.columns:
    console.print(f"Warning: Expected PSAR columns not found. Available: {psar_columns}")
    # Try to find columns with PSARl and PSARs prefixes
    psar_l_candidates = [col for col in psar_ta.columns if col.startswith('PSARl_')]
    psar_s_candidates = [col for col in psar_ta.columns if col.startswith('PSARs_')]
    
    if psar_l_candidates and psar_s_candidates:
        psar_l_col = psar_l_candidates[0]
        psar_s_col = psar_s_candidates[0]
        console.print(f"Using alternative columns: {psar_l_col} and {psar_s_col}")

# Combine PSAR long and short columns to create a single PSAR column as done in our bots
df['psar'] = np.nan
df['psar_direction'] = np.nan

# Merge the PSAR long and short columns
if psar_l_col in psar_ta.columns and psar_s_col in psar_ta.columns:
    # For each row, use PSARl when it's not NaN, otherwise use PSARs
    df['psar'] = psar_ta[psar_l_col].fillna(psar_ta[psar_s_col])
    
    # Set PSAR direction based on PSAR value vs close price
    df['psar_direction'] = np.where(df['psar'] < df['close'], "Bullish", "Bearish")
    
    # Count NaN values in PSAR columns for analysis
    nan_counts = {
        psar_l_col: psar_ta[psar_l_col].isna().sum(),
        psar_s_col: psar_ta[psar_s_col].isna().sum()
    }
    console.print(f"NaN counts in PSAR: {nan_counts}")
    
    # Check for remaining NaN values in combined PSAR column
    nan_count = df['psar'].isna().sum()
    console.print(f"Found {nan_count} NaN values in combined PSAR column")
    
    # Fill any remaining NaN values
    if nan_count > 0:
        df['psar'] = df['psar'].fillna(method='ffill').fillna(method='bfill')
        console.print("All NaN values in PSAR fixed")
else:
    console.print("[red]Critical error: PSAR columns not found in pandas_ta output[/red]")

# Count of bullish and bearish signals
bullish_count = (df['psar_direction'] == "Bullish").sum()
bearish_count = (df['psar_direction'] == "Bearish").sum()
console.print(f"Bullish (PSAR below price) count: {bullish_count}")
console.print(f"Bearish (PSAR above price) count: {bearish_count}")

# Print the first and last few rows of PSAR values for analysis
console.print("\nFirst 5 rows of PSAR data:")
console.print(psar_ta.head())

console.print("\nLast 5 rows of PSAR data:")
console.print(psar_ta.tail())

console.print("\nCombined PSAR and Direction (first 10 rows):")
console.print(df[['close', 'psar', 'psar_direction']].head(10))

# Plot the price and PSAR for visual verification
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Close Price')
plt.scatter(df.index[df['psar_direction'] == "Bullish"], 
            df['psar'][df['psar_direction'] == "Bullish"], 
            color='green', marker='^', label='Bullish PSAR')
plt.scatter(df.index[df['psar_direction'] == "Bearish"], 
            df['psar'][df['psar_direction'] == "Bearish"], 
            color='red', marker='v', label='Bearish PSAR')
plt.title('ETH Price with PSAR Indicator')
plt.xlabel('Candle Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('psar_verification.png')
plt.close()

console.print("\nPSAR visualization saved to 'psar_verification.png'")

# Implement a manual PSAR calculation to verify results
def calculate_manual_psar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    """
    Manual implementation of PSAR calculation for verification
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Initialize arrays
    psar = np.zeros_like(close)
    direction = np.zeros_like(close)  # 1 for bullish, -1 for bearish
    ep = np.zeros_like(close)  # Extreme point
    af = np.zeros_like(close)  # Acceleration factor
    
    # Initialize with a bearish trend (PSAR above price)
    direction[0] = -1
    psar[0] = high[0]
    ep[0] = low[0]
    af[0] = af_start
    
    # Calculate PSAR values
    for i in range(1, len(close)):
        # If previous trend was bullish
        if direction[i-1] == 1:
            # PSAR = Previous PSAR + Previous AF * (Previous EP - Previous PSAR)
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            # Make sure PSAR is below the low of the previous two candles
            psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            
            # Check if trend reversed
            if high[i] >= psar[i]:
                direction[i] = 1  # Still bullish
                # Update extreme point if needed
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    # Increase acceleration factor if we make a new high
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
            else:
                # Trend reversed to bearish
                direction[i] = -1
                psar[i] = ep[i-1]  # Set PSAR to the last extreme point
                ep[i] = low[i]     # Set new extreme point to current low
                af[i] = af_start   # Reset acceleration factor
        
        # If previous trend was bearish
        else:
            # PSAR = Previous PSAR - Previous AF * (Previous PSAR - Previous EP)
            psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
            
            # Make sure PSAR is above the high of the previous two candles
            psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            
            # Check if trend reversed
            if low[i] <= psar[i]:
                direction[i] = -1  # Still bearish
                # Update extreme point if needed
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    # Increase acceleration factor if we make a new low
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
            else:
                # Trend reversed to bullish
                direction[i] = 1
                psar[i] = ep[i-1]  # Set PSAR to the last extreme point
                ep[i] = high[i]    # Set new extreme point to current high
                af[i] = af_start   # Reset acceleration factor
    
    return psar, direction

# Calculate manual PSAR
manual_psar, manual_direction = calculate_manual_psar(df, af_start=psar_step, af_step=psar_step, af_max=psar_max)

# Add manual PSAR to dataframe
df['manual_psar'] = manual_psar
df['manual_direction'] = np.where(manual_direction == 1, "Bullish", "Bearish")

# Compare the results
console.print("\nManual PSAR Calculation (first 10 rows):")
console.print(df[['close', 'psar', 'psar_direction', 'manual_psar', 'manual_direction']].head(10))

# Calculate differences
df['psar_diff'] = df['psar'] - df['manual_psar']
max_diff = df['psar_diff'].abs().max()
console.print(f"\nMaximum difference between pandas_ta and manual PSAR: {max_diff:.6f}")

# Check for direction mismatches
direction_mismatches = (df['psar_direction'] != df['manual_direction']).sum()
console.print(f"Direction mismatches: {direction_mismatches} out of {len(df)} candles")

# Plot comparison between pandas_ta and manual PSAR
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Close Price')
plt.plot(df.index, df['psar'], 'g--', label='pandas_ta PSAR')
plt.plot(df.index, df['manual_psar'], 'r:', label='Manual PSAR')
plt.title('Comparison of PSAR Calculations')
plt.xlabel('Candle Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('psar_comparison.png')
plt.close()

console.print("\nPSAR comparison visualization saved to 'psar_comparison.png'")

# Final conclusion
if max_diff < 0.01 and direction_mismatches == 0:
    console.print("[green]PSAR calculation appears accurate! Both methods give very similar results.[/green]")
elif max_diff < 1.0 and direction_mismatches < len(df) * 0.05:  # Less than 5% direction mismatches
    console.print("[yellow]PSAR calculation has minor discrepancies, but direction signals are mostly consistent.[/yellow]")
else:
    console.print("[red]PSAR calculation shows significant discrepancies! The implementation may need review.[/red]") 