import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# SARIMA modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss , grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse, meanabs
import itertools

class ElectricityPriceForecaster:
    def __init__(self, filter_id="4169", region_id="de",OUTPUT_DIR=None):
        """
        Initialize the forecaster with SMARD API parameters
        
        Parameters:
        - filter_id: SMARD filter ID for prices (4169 for electricity prices)
        - region_id: Region ID (de for Germany)
        """
        self.filter_id = filter_id
        self.region_id = region_id
        self.base_url = "https://www.smard.de/app"
        self.price_data = None
        self.model = None
        self.forecast = None
        self.OUTPUT_DIR=Path(OUTPUT_DIR)

    def fetch_smard_data(self, filter_id, start_date="2024-01-01", end_date="2025-12-31", 
                        resolution="day", data_type="price"):
        """
        Generic function to fetch data from SMARD API
        
        Parameters:
        - filter_id: SMARD filter ID
        - start_date: Start date in 'YYYY-MM-DD' format
        - end_date: End date in 'YYYY-MM-DD' format
        - resolution: 'day' for daily, 'hour' for hourly
        - data_type: 'price' or 'generation'
        """
        print(f"Fetching {data_type} data for filter {filter_id} from {start_date} to {end_date}...")
        
        # First, get indices to find available timestamps
        index_url = f"{self.base_url}/chart_data/{filter_id}/{self.region_id}/index_{resolution}.json"
        response = requests.get(index_url)
        
        if response.status_code != 200:
            print(f"  Error fetching indices for filter {filter_id}: {response.status_code}")
            return None
            
        indices = response.json()
        if 'timestamps' not in indices or not indices['timestamps']:
            print(f"  No timestamps available for filter {filter_id}")
            return None
            
        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Find timestamps within our date range
        target_timestamps = []
        for ts_ms in indices['timestamps']:
            ts_dt = datetime.fromtimestamp(ts_ms / 1000)
            if start_dt <= ts_dt <= end_dt:
                target_timestamps.append(ts_ms)
        
        print(f"  Found {len(target_timestamps)} {resolution}ly timestamps in range")
        
        # Fetch data for each timestamp (limit to avoid too many requests)
        all_data = []
        max_timestamps = 50 if resolution == "hour" else 20
        for ts_ms in target_timestamps[:min(max_timestamps, len(target_timestamps))]:
            data_url = f"{self.base_url}/chart_data/{filter_id}/{self.region_id}/{filter_id}_{self.region_id}_{resolution}_{ts_ms}.json"
            data_response = requests.get(data_url)
            
            if data_response.status_code == 200:
                data = data_response.json()
                if 'series' in data:
                    for point in data['series']:
                        if len(point) == 2 and point[1] is not None:
                            dt = datetime.fromtimestamp(point[0] / 1000)
                            if start_dt <= dt <= end_dt:
                                all_data.append({
                                    'date': dt.date(),
                                    'datetime': dt,
                                    'value': point[1],
                                    'timestamp': point[0]
                                })
        
        if not all_data:
            print(f"  No {data_type} data collected for filter {filter_id}")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(all_data)
        df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
        
        print(f"  Collected {len(df)} data points")
        print(f"  Value range: {df['value'].min():.2f} - {df['value'].max():.2f}")
        
        return df
        
    def fetch_price_data(self, start_date="2024-01-01", end_date="2025-12-31"):
        """
        Fetch daily electricity prices from SMARD API
        
        Parameters:
        - start_date: Start date in 'YYYY-MM-DD' format
        - end_date: End date in 'YYYY-MM-DD' format
        """
        print(f"Fetching price data from {start_date} to {end_date}...")
        
        # First, get indices to find available timestamps
        index_url = f"{self.base_url}/chart_data/{self.filter_id}/{self.region_id}/index_day.json"
        response = requests.get(index_url)
        
        if response.status_code != 200:
            print(f"Error fetching indices: {response.status_code}")
            return None
            
        indices = response.json()
        if 'timestamps' not in indices or not indices['timestamps']:
            print("No timestamps available")
            return None
            
        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Find timestamps within our date range
        target_timestamps = []
        for ts_ms in indices['timestamps']:
            ts_dt = datetime.fromtimestamp(ts_ms / 1000)
            if start_dt <= ts_dt <= end_dt:
                target_timestamps.append(ts_ms)
        
        print(f"Found {len(target_timestamps)} daily timestamps in range")
        
        # Fetch data for each timestamp
        all_data = []
        for ts_ms in target_timestamps:
            data_url = f"{self.base_url}/chart_data/{self.filter_id}/{self.region_id}/{self.filter_id}_{self.region_id}_day_{ts_ms}.json"
            data_response = requests.get(data_url)
            
            if data_response.status_code == 200:
                data = data_response.json()
                if 'series' in data:
                    for point in data['series']:
                        if len(point) == 2 and point[1] is not None:
                            dt = datetime.fromtimestamp(point[0] / 1000)
                            # Only include data points between start and end dates
                            if start_dt <= dt <= end_dt:
                                all_data.append({
                                    'date': dt.date(),
                                    'datetime': dt,
                                    'price': point[1],
                                    'timestamp': point[0]
                                })
        
        if not all_data:
            print("No data collected")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(all_data)
        df = df.sort_values('datetime').drop_duplicates('date').reset_index(drop=True)
        
        # Ensure we have 730 days (approx 2 years)
        df = df.head(730)
        
        print(f"Collected {len(df)} days of price data")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Average price: {df['price'].mean():.2f} EUR/MWh")
        print(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f} EUR/MWh")
        
        self.price_data = df.set_index('datetime')['price']
        return self.price_data
    
    def fetch_exogenous_variables(self, start_date="2024-01-01", end_date="2025-12-31"):
        """
        Fetch exogenous variables based on specified filters
        
        Parameters (as per your request):
        - Solar: filter = 4068
        - Wind: filter = 1225  
        - Coal: filters = 4069, 1223
        - Gas: filter = 4071
        """
        print("\n" + "="*70)
        print("FETCHING EXOGENOUS VARIABLES")
        print("="*70)
        
        # Define exogenous variables (filter_id: name)
        exog_vars = {
            "4068": "solar",      # Solar production
            "1225": "wind",       # Wind production
            "4069": "coal_1",     # Coal production (first filter)
            "1223": "coal_2",     # Coal production (second filter)
            "4071": "gas",        # Gas production
        }
        
        all_exog_data = {}
        
        for filter_id, var_name in exog_vars.items():
            print(f"\nFetching {var_name} data (Filter: {filter_id})...")
            
            df = self.fetch_smard_data(
                filter_id=filter_id,
                start_date=start_date,
                end_date=end_date,
                resolution="day",
                data_type="generation"
            )
            
            if df is not None:
                # Resample to daily frequency
                df_daily = df.set_index('datetime')['value'].resample('D').mean()
                all_exog_data[var_name] = df_daily.dropna()
                
                print(f"  {var_name}: {len(df_daily)} days, Avg: {df_daily.mean():.1f} MW")
        
        # Combine all exogenous variables
        if all_exog_data:
            # Align all series to common index (intersection)
            exog_series = []
            common_idx = None
            
            for var_name, series in all_exog_data.items():
                if common_idx is None:
                    common_idx = series.index
                else:
                    common_idx = common_idx.intersection(series.index)
            
            # Create DataFrame with aligned data
            exog_df = pd.DataFrame(index=common_idx)
            
            for var_name, series in all_exog_data.items():
                exog_df[var_name] = series.reindex(common_idx)
            
            # Handle missing values
            exog_df = exog_df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"\nCombined Exogenous Variables:")
            print(f"  Common date range: {exog_df.index.min()} to {exog_df.index.max()}")
            print(f"  Variables: {list(exog_df.columns)}")
            print(f"  Total days: {len(exog_df)}")
            
            # Add combined coal variable (sum of both coal filters)
            if 'coal_1' in exog_df.columns and 'coal_2' in exog_df.columns:
                exog_df['coal_total'] = exog_df['coal_1'] + exog_df['coal_2']
                print(f"  Added 'coal_total' (sum of coal_1 and coal_2)")
            
            self.exog_data = exog_df
            
            # Analyze correlation with prices
            if self.price_data is not None:
                self.analyze_exog_correlations(exog_df)
        
        return self.exog_data
    
    
    def test_stationarity(self, series, title="Time Series"):
        """
        Perform stationarity tests (ADF and KPSS) as described in the paper
        
        Returns:
        - True if series is stationary, False otherwise
        """
        print(f"\n=== Stationarity Test for {title} ===")
        
        # ADF Test (null: non-stationary)
        adf_result = adfuller(series.dropna())
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"ADF p-value: {adf_result[1]:.4f}")
        
        # KPSS Test (null: stationary)
        kpss_result = kpss(series.dropna(), regression='c')
        print(f"KPSS Statistic: {kpss_result[0]:.4f}")
        print(f"KPSS p-value: {kpss_result[1]:.4f}")
        
        # Determine stationarity
        is_stationary = (adf_result[1] < 0.05) and (kpss_result[1] > 0.05)
        
        if is_stationary:
            print("✓ Series is stationary")
        else:
            print("✗ Series is non-stationary")
            
        return is_stationary
    
    def prepare_data(self, p=1, d=1, q=2, P=2, D=2, Q=2, S=7):
        """
        Prepare data for SARIMA modeling
        
        Parameters:
        - p: AR order
        - d: Differencing order
        - q: MA order
        - P: Seasonal AR order
        - D: Seasonal differencing order
        - Q: Seasonal MA order
        - S: Seasonal period (7 for weekly)
        """
        if self.price_data is None:
            print("No price data available. Fetch data first.")
            return None
            
        print("\n=== Data Preparation ===")
        
        # 1. Test original series for stationarity
        original_stationary = self.test_stationarity(self.price_data, "Original Prices")
        
        # 2. Apply differencing if needed (based on paper methodology)
        if d > 0 or not original_stationary:
            print(f"\nApplying {d} non-seasonal differencing...")
            diff_series = self.price_data.diff(d).dropna()
            
            # Test differenced series
            diff_stationary = self.test_stationarity(diff_series, f"Differenced (d={d}) Prices")
            
            if not diff_stationary and d == 1:
                print("Series still non-stationary after differencing.")
                print("Consider increasing d or applying seasonal differencing.")
        
        # 3. Plot ACF and PACF for model identification (as done in paper)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original series
        plot_acf(self.price_data, lags=40, ax=axes[0, 0])
        axes[0, 0].set_title('ACF - Original Prices')
        
        plot_pacf(self.price_data, lags=40, ax=axes[0, 1])
        axes[0, 1].set_title('PACF - Original Prices')
        
        # Differenced series
        if d > 0:
            diff_series = self.price_data.diff(d).dropna()
            plot_acf(diff_series, lags=40, ax=axes[1, 0])
            axes[1, 0].set_title(f'ACF - Differenced (d={d})')
            
            plot_pacf(diff_series, lags=40, ax=axes[1, 1])
            axes[1, 1].set_title(f'PACF - Differenced (d={d})')
        
        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR /'acf_pacf_plots.png', dpi=150)
        plt.show()
        
        return self.price_data
    
    def build_sarima_model(self, p=1, d=1, q=2, P=2, D=2, Q=2, S=7):
        """
        Build SARIMA model based on paper methodology
        
        Parameters (from paper):
        - SARIMA(1,1,2)(2,2,7) was found optimal in the paper
        - You can adjust these parameters
        """
        print(f"\n=== Building SARIMA({p},{d},{q})({P},{D},{Q})_{S} Model ===")
        
        if self.price_data is None:
            print("No data available. Fetch data first.")
            return None
        
        try:
            # Split data into training set (as in paper: use historical data to predict future)
            train_data = self.price_data
            
            # Build SARIMA model (without exogenous variables as requested)
            self.model = SARIMAX(
                train_data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit the model
            print("Fitting model...")
            self.model_fit = self.model.fit(method="lbfgs",maxiter=300,disp=False)
            
            print("Model fitting complete!")
            print(self.model_fit.summary())
            print("MLE retvals")
            print(self.model_fit.mle_retvals)

            # Save model summary to file
            with open(self.OUTPUT_DIR / 'sarima_model_summary.txt', 'w') as f:
                f.write(str(self.model_fit.summary()))
            
            return self.model_fit
            
        except Exception as e:
            print(f"Error building model: {e}")
            return None
    
    def analyze_exog_correlations(self, exog_df):
        """
        Analyze correlation between exogenous variables and prices
        """
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS WITH PRICES")
        print("="*70)
        
        # Align price and exogenous data
        aligned_data = pd.DataFrame(index=exog_df.index)
        aligned_data['price'] = self.price_data.reindex(exog_df.index)
        
        for col in exog_df.columns:
            aligned_data[col] = exog_df[col]
        
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) > 0:
            # Calculate correlations
            correlations = {}
            for col in exog_df.columns:
                corr = aligned_data['price'].corr(aligned_data[col])
                correlations[col] = corr
            
            # Sort by absolute correlation
            sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("\nCorrelation with electricity prices:")
            for var, corr in sorted_corrs:
                direction = "positive" if corr > 0 else "negative"
                print(f"  {var:15} : {corr:+.3f} ({direction} relationship)")
            
            # Create correlation plot
            self.plot_correlations(aligned_data)
            
            # Granger causality tests (simplified version)
            print("\nGranger Causality (lag 1):")
            for var in exog_df.columns[:3]:  # Test first 3 variables
                test_data = aligned_data[['price', var]].dropna()
                if len(test_data) > 50:
                    try:
                        gc_test = grangercausalitytests(test_data, maxlag=1, verbose=False)
                        p_value = gc_test[1][0]['ssr_ftest'][1]
                        if p_value < 0.05:
                            print(f"  {var:15} : Granger-causes prices (p={p_value:.3f})")
                        else:
                            print(f"  {var:15} : No Granger causality (p={p_value:.3f})")
                    except:
                        print(f"  {var:15} : Could not perform test")
        
        return aligned_data
    
    def plot_correlations(self, aligned_data):
        """Plot correlation analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        exog_vars = [col for col in aligned_data.columns if col != 'price']
        
        for idx, var in enumerate(exog_vars[:6]):  # Plot first 6 variables
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(aligned_data[var], aligned_data['price'], alpha=0.5, s=20)
            
            # Add regression line
            z = np.polyfit(aligned_data[var], aligned_data['price'], 1)
            p = np.poly1d(z)
            ax.plot(aligned_data[var], p(aligned_data[var]), "r-", linewidth=2)
            
            # Calculate correlation
            corr = aligned_data['price'].corr(aligned_data[var])
            
            ax.set_xlabel(f'{var} (MW)')
            ax.set_ylabel('Price (EUR/MWh)')
            ax.set_title(f'{var} vs Price\nCorrelation: {corr:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR / 'exog_correlations.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def prepare_sarimax_data(self, p=1, d=1, q=2, P=2, D=2, Q=2, S=7):
        """
        Prepare data for SARIMAX modeling with exogenous variables
        """
        if self.price_data is None:
            print("No price data available. Fetch data first.")
            return None
        
        print("\n" + "="*70)
        print("PREPARING SARIMAX DATA")
        print("="*70)
        
        # Test stationarity of price data
        self.test_stationarity(self.price_data, "Original Prices")
        
        # Test stationarity of exogenous variables
        if self.exog_data is not None:
            print("\nTesting stationarity of exogenous variables:")
            for col in self.exog_data.columns:
                self.test_stationarity(self.exog_data[col], f"Exog: {col}")
        
        # Create lagged exogenous variables (1-day lag as in paper)
        if self.exog_data is not None:
            exog_lagged = self.exog_data.shift(1).dropna()
            
            # Align price and exogenous data
            common_idx = self.price_data.index.intersection(exog_lagged.index)
            price_aligned = self.price_data.reindex(common_idx)
            exog_aligned = exog_lagged.reindex(common_idx)
            
            print(f"\nAligned data for modeling:")
            print(f"  Price data points: {len(price_aligned)}")
            print(f"  Exogenous variables: {list(exog_aligned.columns)}")
            print(f"  Date range: {price_aligned.index.min()} to {price_aligned.index.max()}")
            
            return price_aligned, exog_aligned
        
        return self.price_data, None
    
    def build_sarimax_model(self, p=1, d=1, q=1, P=1, D=1, Q=1, S=7, exog_vars=None):
        """
        Build SARIMAX model with exogenous variables
        
        Parameters:
        - exog_vars: list of exogenous variable names to include
        """
        print(f"\n" + "="*70)
        print(f"BUILDING SARIMAX({p},{d},{q})({P},{D},{Q})_{S} MODEL")
        print("="*70)
        
        if self.price_data is None:
            print("No price data available. Fetch data first.")
            return None
        
        # Prepare data
        price_aligned, exog_aligned = self.prepare_sarimax_data(p, d, q, P, D, Q, S)
        
        if price_aligned is None:
            print("Could not prepare data for modeling")
            return None
        
        # Select exogenous variables
        if exog_aligned is not None and exog_vars is not None:
            # Use specified exogenous variables
            selected_exog = exog_aligned[exog_vars]
            print(f"\nSelected exogenous variables: {list(selected_exog.columns)}")
        elif exog_aligned is not None:
            # Use all exogenous variables
            selected_exog = exog_aligned
            print(f"\nUsing all exogenous variables: {list(selected_exog.columns)}")
        else:
            selected_exog = None
            print("\nNo exogenous variables available")
        
        try:
            # Split data (80% train, 20% test)
            split_idx = int(len(price_aligned) * 0.8)
            train_endog = price_aligned.iloc[:split_idx]
            test_endog = price_aligned.iloc[split_idx:]
            
            if selected_exog is not None:
                train_exog = selected_exog.iloc[:split_idx]
                test_exog = selected_exog.iloc[split_idx:]
            else:
                train_exog = None
                test_exog = None
            
            print(f"\nTrain-Test Split:")
            print(f"  Train: {len(train_endog)} days ({train_endog.index.min()} to {train_endog.index.max()})")
            print(f"  Test:  {len(test_endog)} days ({test_endog.index.min()} to {test_endog.index.max()})")
            
            # Build SARIMAX model
            self.model = SARIMAX(
                endog=train_endog,
                exog=train_exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit the model
            print("\nFitting SARIMAX model...")
            self.model_fit = self.model.fit(disp=False, maxiter=200)
            
            print("\n✓ Model fitting complete!")
            print(self.model_fit.summary())
            
            # Save model summary
            with open(self.OUTPUT_DIR / 'sarimax_model_summary.txt', 'w') as f:
                f.write(str(self.model_fit.summary()))
            
            # Test set predictions
            if test_exog is not None and len(test_endog) > 0:
                test_predictions = self.model_fit.predict(
                    start=test_endog.index[0],
                    end=test_endog.index[-1],
                    exog=test_exog
                )
                
                # Calculate test metrics
                mape = np.mean(np.abs((test_endog - test_predictions) / test_endog)) * 100
                rmse_val = rmse(test_endog, test_predictions)
                
                print(f"\nTest Set Performance:")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  RMSE: {rmse_val:.2f}")
                
                # Compare with SARIMA without exogenous variables
                self.compare_with_sarima(train_endog, test_endog, p, d, q, P, D, Q, S)
            
            return self.model_fit
            
        except Exception as e:
            print(f"\n✗ Error building model: {e}")
            import traceback
            traceback.print_exc()
            return None
    

    
    def forecast_prices(self, forecast_start="2026-01-01", forecast_end="2026-01-31", 
                       exog_forecast=None):
        """
        Forecast electricity prices with exogenous variables
        
        Parameters:
        - exog_forecast: DataFrame with exogenous variables for forecast period
        """
        if self.model_fit is None:
            print("No model fitted. Build model first.")
            return None
        
        print(f"\n" + "="*70)
        print(f"FORECASTING PRICES FROM {forecast_start} TO {forecast_end}")
        print("="*70)
        
        # Calculate number of steps to forecast
        start_date = datetime.strptime(forecast_start, "%Y-%m-%d")
        end_date = datetime.strptime(forecast_end, "%Y-%m-%d")
        forecast_steps = (end_date - start_date).days + 1
        
        # Create forecast dates
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate forecast
    
        if exog_forecast is not None:
            # Ensure exog_forecast has correct index
            if len(exog_forecast) != forecast_steps:
                print(f"Warning: exog_forecast has {len(exog_forecast)} rows, but need {forecast_steps}")
                exog_forecast = exog_forecast.reindex(forecast_dates).fillna(method='ffill').fillna(method='bfill')
            
            forecast = self.model_fit.get_forecast(
                steps=forecast_steps,
                exog=exog_forecast
            )
        else:
            forecast = self.model_fit.get_forecast(steps=forecast_steps)
        
        forecast_mean = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()
        
        # Set correct index
        forecast_mean.index = forecast_dates
        forecast_conf_int.index = forecast_dates
        
        self.forecast = {
            'dates': forecast_dates,
            'mean': forecast_mean,
            'conf_int': forecast_conf_int,
            'exog_used': exog_forecast is not None
        }
        
        print(f"✓ Generated forecast for {forecast_steps} days")
        print(f"  Forecast price range: {forecast_mean.min():.2f} - {forecast_mean.max():.2f} EUR/MWh")
        print(f"  Average forecast price: {forecast_mean.mean():.2f} EUR/MWh")
        
        if exog_forecast is not None:
            print(f"  Exogenous variables used: {list(exog_forecast.columns)}")
        
        return self.forecast

    def evaluate_forecast(self, actual_prices=None):
        """
        Evaluate forecast accuracy if actual prices are available
        
        Parameters:
        - actual_prices: Pandas Series with actual prices for forecast period
        """
        if self.forecast is None:
            print("No forecast available. Generate forecast first.")
            return None
        
        if actual_prices is not None:
            print("\n=== Forecast Evaluation ===")
            
            # Align actual prices with forecast dates
            actual_aligned = actual_prices.reindex(self.forecast['dates'])
            
            # Calculate error metrics (as done in paper)
            mape = np.mean(np.abs((actual_aligned - self.forecast['mean']) / actual_aligned)) * 100
            rmse_val = rmse(actual_aligned, self.forecast['mean'])
            mae = meanabs(actual_aligned, self.forecast['mean'])
            
            print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
            print(f"RMSE (Root Mean Square Error): {rmse_val:.2f}")
            print(f"MAE (Mean Absolute Error): {mae:.2f}")
            
            # Paper comparison: Their MAPE was 1.95%
            if mape < 10:
                print("✓ Excellent forecast accuracy (MAPE < 10%)")
            elif mape < 20:
                print("✓ Good forecast accuracy (MAPE < 20%)")
            else:
                print("✗ Forecast accuracy could be improved")
            
            return {
                'mape': mape,
                'rmse': rmse_val,
                'mae': mae
            }
        else:
            print("No actual prices provided for evaluation")
            return None
    
    def plot_results(self, actual_prices=None, save_plot=True,exog_forecast=None):
        """
        Plot forecast results with actual prices (if available)
        """
        if self.forecast is None:
            print("No forecast available. Generate forecast first.")
            return
        if exog_forecast is None:
            print("\n=== Generating Forecast Plot ===")
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Full time series with forecast
            ax1 = axes[0]
            
            # Plot historical data
            ax1.plot(self.price_data.index, self.price_data.values, 
                    'b-', linewidth=1.5, label='Historical Prices', alpha=0.8)
            
            # Plot forecast
            ax1.plot(self.forecast['dates'], self.forecast['mean'].values,
                    'r-', linewidth=2, label='SARIMA Forecast')
            
            # # Plot confidence interval
            # ax1.fill_between(self.forecast['dates'],
            #                 self.forecast['conf_int'].iloc[:, 0],
            #                 self.forecast['conf_int'].iloc[:, 1],
            #                 color='red', alpha=0.2, label='95% Confidence Interval')
            
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price (EUR/MWh)')
            ax1.set_title('Electricity Price Forecast: SARIMA Model')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add forecast start line
            forecast_start = self.forecast['dates'][0]
            ax1.axvline(x=forecast_start, color='green', linestyle='--', 
                    alpha=0.7, label='Forecast Start')
            
            # Plot 2: Zoom on forecast period
            ax2 = axes[1]
            
            # Plot forecast
            ax2.plot(self.forecast['dates'], self.forecast['mean'].values,
                    'r-', linewidth=2, label='SARIMA Forecast', marker='o')
            
            # Plot actual prices if available
            if actual_prices is not None:
                actual_aligned = actual_prices.reindex(self.forecast['dates'])
                ax2.plot(self.forecast['dates'], actual_aligned.values,
                        'g-', linewidth=2, label='Actual Prices', marker='s')
                
                # Calculate and display MAPE
                mape = np.mean(np.abs((actual_aligned - self.forecast['mean']) / actual_aligned)) * 100
                ax2.text(0.02, 0.95, f'MAPE: {mape:.2f}%', 
                        transform=ax2.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # # Plot confidence interval
            # ax2.fill_between(self.forecast['dates'],
            #                 self.forecast['conf_int'].iloc[:, 0],
            #                 self.forecast['conf_int'].iloc[:, 1],
            #                 color='red', alpha=0.2, label='95% CI')
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price (EUR/MWh)')
            ax2.set_title('January 2026: Forecast vs Actual (if available)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.OUTPUT_DIR / 'sarima_forecast_plot.png', dpi=150, bbox_inches='tight')
                print("Plot saved as 'sarima_forecast_plot.png'")
            
            plt.show()
        else:
            if self.forecast is None:
                print("No forecast available. Generate forecast first.")
                return
            
            print("\n" + "="*70)
            print("GENERATING FORECAST PLOTS")
            print("="*70)
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 15))
            
            # Plot 1: Full time series with forecast
            ax1 = axes[0]
            
            if self.price_data is not None:
                ax1.plot(self.price_data.index, self.price_data.values, 
                        'b-', linewidth=1.5, label='Historical Prices', alpha=0.7)
            
            ax1.plot(self.forecast['dates'], self.forecast['mean'].values,
                    'r-', linewidth=2, label='SARIMAX Forecast')
            
            #ax1.fill_between(self.forecast['dates'],
            #               self.forecast['conf_int'].iloc[:, 0],
            #              self.forecast['conf_int'].iloc[:, 1],
            #             color='red', alpha=0.2, label='95% Confidence Interval')
            
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price (EUR/MWh)')
            ax1.set_title('Electricity Price Forecast: SARIMAX Model with Exogenous Variables')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            if self.price_data is not None:
                forecast_start = self.forecast['dates'][0]
                ax1.axvline(x=forecast_start, color='green', linestyle='--', 
                        alpha=0.7, label='Forecast Start')
            
            # Plot 2: Zoom on forecast period
            ax2 = axes[1]
            
            ax2.plot(self.forecast['dates'], self.forecast['mean'].values,
                    'r-', linewidth=2, label='SARIMAX Forecast', marker='o')
            
            if actual_prices is not None:
                actual_aligned = actual_prices.reindex(self.forecast['dates'])
                ax2.plot(self.forecast['dates'], actual_aligned.values,
                        'g-', linewidth=2, label='Actual Prices', marker='s')
                
                mape = np.mean(np.abs((actual_aligned - self.forecast['mean']) / actual_aligned)) * 100
                ax2.text(0.02, 0.95, f'MAPE: {mape:.2f}%', 
                        transform=ax2.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # ax2.fill_between(self.forecast['dates'],
            #                 self.forecast['conf_int'].iloc[:, 0],
            #                 self.forecast['conf_int'].iloc[:, 1],
            #                 color='red', alpha=0.2, label='95% CI')
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price (EUR/MWh)')
            ax2.set_title('January 2026: Forecast vs Actual (if available)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Exogenous variables impact (if available)
            ax3 = axes[2]
            
            if self.exog_data is not None and hasattr(self, 'model_fit'):
                # Get model coefficients for exogenous variables
                exog_coeffs = {}
                for param in self.model_fit.params.index:
                    if any(exog_var in param for exog_var in self.exog_data.columns):
                        exog_coeffs[param] = self.model_fit.params[param]
                
                if exog_coeffs:
                    variables = list(exog_coeffs.keys())
                    coefficients = list(exog_coeffs.values())
                    
                    colors = ['red' if coef < 0 else 'green' for coef in coefficients]
                    
                    bars = ax3.barh(variables, coefficients, color=colors, edgecolor='black')
                    
                    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    ax3.set_xlabel('Coefficient Value')
                    ax3.set_title('Impact of Exogenous Variables on Price')
                    ax3.grid(True, alpha=0.3, axis='x')
                    
                    # Add coefficient values on bars
                    for bar, coef in zip(bars, coefficients):
                        width = bar.get_width()
                        ax3.text(width if width > 0 else width - 0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{coef:.4f}', ha='left' if width > 0 else 'right',
                                va='center', fontsize=9)
                else:
                    ax3.text(0.5, 0.5, 'No exogenous variable coefficients available',
                            ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'Exogenous variable analysis not available',
                        ha='center', va='center', transform=ax3.transAxes)
            
            plt.tight_layout()
            
            if save_plot:
                filename = 'sarimax_forecast_plot.png'
                plt.savefig(self.OUTPUT_DIR/filename, dpi=150, bbox_inches='tight')
                print(f"✓ Plot saved as '{filename}'")
            
            plt.show()
            
            # Save forecast to CSV
            forecast_df = pd.DataFrame({
                'date': self.forecast['dates'],
                'forecast_price': self.forecast['mean'].values,
                'lower_ci': self.forecast['conf_int'].iloc[:, 0].values,
                'upper_ci': self.forecast['conf_int'].iloc[:, 1].values
            })
            
            forecast_df.to_csv(self.OUTPUT_DIR / 'sarimax_forecast_jan_2026.csv', index=False)
            print(f"✓ Forecast saved to 'sarimax_forecast_jan_2026.csv'")
        
        # Also create histogram of forecast errors (as in paper Figure 3)
        if actual_prices is not None:
            fig2, ax3 = plt.subplots(figsize=(10, 6))
            
            errors = (actual_aligned - self.forecast['mean']).dropna()
            
            ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            ax3.axvline(x=errors.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
            
            ax3.set_xlabel('Forecast Error (EUR/MWh)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Histogram of Forecast Errors (as in Paper Fig. 3)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.OUTPUT_DIR / 'forecast_errors_histogram.png', dpi=150)
            plt.show()

    def create_exog_forecast_scenarios(self, forecast_dates):
        """
        Create different scenarios for exogenous variables during forecast period
        
        Returns:
        - Dictionary with different scenarios for exogenous variables
        """
        if self.exog_data is None:
            print("No historical exogenous data available for scenario creation")
            return None
        
        print("\n" + "="*70)
        print("CREATING EXOGENOUS VARIABLE SCENARIOS")
        print("="*70)
        
        scenarios = {}
        
        # Scenario 1: Historical average (baseline)
        historical_avg = self.exog_data.mean()
        scenarios['baseline'] = pd.DataFrame(
            [historical_avg] * len(forecast_dates),
            index=forecast_dates,
            columns=self.exog_data.columns
        )
        
        # Scenario 2: Same as last year same period
        last_year_dates = forecast_dates - pd.DateOffset(years=1)
        last_year_mask = last_year_dates.isin(self.exog_data.index)
        
        if last_year_mask.any():
            scenarios['last_year'] = pd.DataFrame(index=forecast_dates)
            for col in self.exog_data.columns:
                scenarios['last_year'][col] = self.exog_data[col].reindex(last_year_dates).values
        
        # Scenario 3: Seasonal pattern (weekly average)
        day_of_week = forecast_dates.dayofweek
        weekly_avg = self.exog_data.groupby(self.exog_data.index.dayofweek).mean()
        
        scenarios['seasonal'] = pd.DataFrame(index=forecast_dates)
        for col in self.exog_data.columns:
            scenarios['seasonal'][col] = weekly_avg[col].reindex(day_of_week).values
        
        print(f"Created {len(scenarios)} scenarios:")
        for name, df in scenarios.items():
            print(f"  {name}: {df.shape[0]} days, {df.shape[1]} variables")
        
        return scenarios

    def grid_search_parameters(self, p_range=range(0, 3), d_range=range(0, 2), 
                              q_range=range(0, 3), P_range=range(0, 3),
                              D_range=range(0, 2), Q_range=range(0, 3),
                              S=7):
        """
        Perform grid search for optimal SARIMA parameters using AIC
        (as done in the paper, Table II)
        """
        print("\n=== Performing Grid Search for Optimal Parameters ===")
        print("This may take several minutes...")
        
        best_aic = float('inf')
        best_params = None
        
        # Create all combinations of parameters
        param_combinations = list(itertools.product(p_range, d_range, q_range,
                                                   P_range, D_range, Q_range))
        
        results = []
        
        for i, (p, d, q, P, D, Q) in enumerate(param_combinations):
            try:
                # Skip if both p and q are 0
                if p == 0 and q == 0:
                    continue
                    
                # Fit model
                model = SARIMAX(self.price_data,
                              order=(p, d, q),
                              seasonal_order=(P, D, Q, S),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                
                model_fit = model.fit(disp=False)
                aic = model_fit.aic
                
                results.append({
                    'p': p, 'd': d, 'q': q,
                    'P': P, 'D': D, 'Q': Q,
                    'AIC': aic
                })
                
                # Update best model
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q, P, D, Q)
                
                if (i + 1) % 10 == 0:
                    print(f"  Tested {i + 1}/{len(param_combinations)} combinations...")
                    
            except Exception as e:
                continue
        
        # Sort results by AIC
        results_df = pd.DataFrame(results).sort_values('AIC')
        
        print("\n=== Top 5 Parameter Combinations by AIC ===")
        print(results_df.head())
        
        print(f"\n✓ Best parameters: SARIMA{best_params[:3]}{best_params[3:]}_{S}")
        print(f"✓ Best AIC: {best_aic:.2f}")
        
        # Save results to CSV
        results_df.to_csv(self.OUTPUT_DIR / 'sarima_grid_search_results.csv', index=False)
        print("Grid search results saved to 'sarima_grid_search_results.csv'")
        
        return best_params, results_df
    pass
