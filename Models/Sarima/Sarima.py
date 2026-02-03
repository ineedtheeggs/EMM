"""
SARIMA Model for Electricity Price Forecasting
Based on: "A Seasonal ARIMA Model With Exogenous Variables for Elspot Electricity Prices in Sweden" by Xie et al.
"""
from EnergyForecasting.EnergyForecasting import ElectricityPriceForecaster as epf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("SARIMA MODEL FOR ELECTRICITY PRICE FORECASTING")
print("Based on: 'A Seasonal ARIMA Model With Exogenous Variables")
print("           for Elspot Electricity Prices in Sweden' by Xie et al.")
print("="*70)

# Initialize forecaster
forecaster = epf(filter_id="4169", region_id="de", OUTPUT_DIR=OUTPUT_DIR)

# Step 1: Fetch historical price data (730 days from 2024-01-01 to 2025-12-31)
price_data = forecaster.fetch_price_data(start_date="2024-01-01", 
                                        end_date="2025-12-31")

if price_data is None:
    print("Failed to fetch price data. Exiting.")


# Step 2: Prepare data and analyze stationarity
forecaster.prepare_data()

# Step 3: Option 1: Use optimal parameters from paper
print("\n" + "="*70)
print("OPTION 1: Using parameters from paper: SARIMA(1,1,2)(2,2,7)")
print("="*70)

# Build model with paper's optimal parameters
model_fit = forecaster.build_sarima_model(p=1, d=1, q=1, P=1, D=1, Q=1, S=7)

# Step 4: Generate forecast for January 2026
forecast = forecaster.forecast_prices(forecast_start="2026-01-01",
                                        forecast_end="2026-01-31")

# Step 5: Try to fetch actual prices for January 2026 for evaluation
print("\nAttempting to fetch actual prices for January 2026 for evaluation...")
try:
    # Note: This will only work if 2026 data is available in SMARD
    actual_prices_2026 = forecaster.fetch_price_data(start_date="2026-01-01",
                                                        end_date="2026-01-31")
except:
    print("Actual 2026 prices not available yet. Plotting forecast only.")
    actual_prices_2026 = None

# Step 6: Plot results
forecaster.plot_results(actual_prices=actual_prices_2026)


print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("1. acf_pacf_plots.png - ACF/PACF plots for model identification")
print("2. sarima_forecast_plot.png - Main forecast plot")
print("3. forecast_errors_histogram.png - Error distribution (if actual data available)")
print("4. sarima_model_summary.txt - Detailed model summary")
print("5. sarima_grid_search_results.csv - Grid search results (if performed)")

print("\nTo use different SARIMA parameters, modify the build_sarima_model() call.")
print("Example: forecaster.build_sarima_model(p=2, d=1, q=1, P=1, D=1, Q=1, S=7)")
