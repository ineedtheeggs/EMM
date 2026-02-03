"""
SARIMAX Model for Electricity Price Forecasting
Based on: "A Seasonal ARIMA Model With Exogenous Variables for Elspot Electricity Prices in Sweden" by Xie et al.
"""
from EnergyForecasting.EnergyForecasting import ElectricityPriceForecaster as epf
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

"""
Main execution function
"""
print("="*70)
print("SARIMAX MODEL FOR ELECTRICITY PRICE FORECASTING")
print("With Exogenous Variables (Solar, Wind, Coal, Gas)")
print("Based on: 'A Seasonal ARIMA Model With Exogenous Variables")
print("           for Elspot Electricity Prices in Sweden' by Xie et al.")
print("="*70)

# Initialize forecaster
forecaster = epf(filter_id="4169", region_id="de",OUTPUT_DIR=OUTPUT_DIR)

# Step 1: Fetch historical price data (2023-01-01 to 2025-12-31)
print("\n" + "="*70)
print("STEP 1: FETCHING PRICE DATA")
print("="*70)
price_data = forecaster.fetch_price_data(
    start_date="2023-01-01", 
    end_date="2025-12-31"
)

if price_data is None:
    print("Failed to fetch price data. Exiting.")
    

# Step 2: Fetch exogenous variables
print("\n" + "="*70)
print("STEP 2: FETCHING EXOGENOUS VARIABLES")
print("="*70)
exog_data = forecaster.fetch_exogenous_variables(
    start_date="2023-01-01",
    end_date="2025-12-31"
)

# Step 3: Build SARIMAX model with selected exogenous variables
print("\n" + "="*70)
print("STEP 3: BUILDING SARIMAX MODEL")
print("="*70)

# Select which exogenous variables to use (based on correlation analysis)
if exog_data is not None:
    # Use variables with strongest correlation (you can adjust this)
    exog_vars_to_use = ['wind', 'solar', 'coal_total', 'gas']
    # Filter to only include variables that exist in the data
    exog_vars_to_use = [var for var in exog_vars_to_use if var in exog_data.columns]
else:
    exog_vars_to_use = None

# Build model using parameters from paper
model_fit = forecaster.build_sarimax_model(
    p=1, d=1, q=1,          # Non-seasonal parameters
    P=1, D=1, Q=1, S=7,     # Seasonal parameters
    exog_vars=exog_vars_to_use
)

if model_fit is None:
    print("Failed to build model. Exiting.")


# Step 4: Create forecast scenarios for January 2026
print("\n" + "="*70)
print("STEP 4: CREATING FORECAST")
print("="*70)

forecast_dates = pd.date_range(start="2026-01-01", end="2026-01-31", freq='D')

if exog_data is not None:
    # Create exogenous variable scenarios for forecast period
    scenarios = forecaster.create_exog_forecast_scenarios(forecast_dates)
    
    # Use baseline scenario for forecast
    if scenarios and 'baseline' in scenarios:
        exog_forecast = scenarios['baseline']
        print(f"\nUsing 'baseline' scenario for exogenous variables")
    else:
        # Use historical average
        exog_forecast = pd.DataFrame(
            [exog_data.mean()] * len(forecast_dates),
            index=forecast_dates,
            columns=exog_data.columns
        )
else:
    exog_forecast = None

# Generate forecast
forecast = forecaster.forecast_prices(
    forecast_start="2026-01-01",
    forecast_end="2026-01-31",
    exog_forecast=exog_forecast[exog_vars_to_use] if exog_forecast is not None and exog_vars_to_use else None
)

# Step 5: Try to fetch actual prices for evaluation
print("\n" + "="*70)
print("STEP 5: EVALUATION (IF DATA AVAILABLE)")
print("="*70)

actual_prices_2026 = None
try:
    actual_prices_2026 = forecaster.fetch_price_data(
        start_date="2026-01-01",
        end_date="2026-01-31"
    )
except:
    print("Actual 2026 prices not available yet for evaluation.")

# Step 6: Plot results
forecaster.plot_results(actual_prices=actual_prices_2026,exog_forecast=exog_forecast[exog_vars_to_use] if exog_forecast is not None and exog_vars_to_use else None)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("1. exog_correlations.png - Correlation plots")
print("2. model_comparison.png - SARIMA vs SARIMAX comparison")
print("3. sarimax_forecast_plot.png - Main forecast plot")
print("4. sarimax_forecast_jan_2026.csv - Forecast data")
print("5. sarimax_model_summary.txt - Detailed model summary")

print("\nKey Insights:")
print("- SARIMAX includes exogenous variables: solar, wind, coal, gas")
print("- Uses 1-day lagged exogenous values (as in Xie et al. paper)")
print("- Model shows impact of each generation type on prices")
