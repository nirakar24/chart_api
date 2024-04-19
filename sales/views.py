import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Sales

@csrf_exempt
def forecast_sales(request, product_id, csv_file=None):
    # Convert product_id to integer
    product_id = int(product_id)
    
    # Initialize dictionary to store data for the product
    product_data = {}
    
    # Load dataset
    if csv_file:
        df = pd.read_csv(csv_file)
    else:
        # Fetch data from Sales model
        sales_data = Sales.objects.filter(product=product_id).values('date', 'sales')
        df = pd.DataFrame(list(sales_data))

    # Filter dataset for the specified product
    product_df = df.copy()

    # Prepare data for Prophet
    product_df['date'] = pd.to_datetime(product_df['date'])
    product_df = product_df.rename(columns={'date': 'ds', 'sales': 'y'})

    # Instantiate Prophet model
    model = Prophet()

    # Fit model
    model.fit(product_df)

    # Make future dataframe for forecasting
    future = model.make_future_dataframe(periods=365)

    # Forecast
    forecast = model.predict(future)

    # Define future timeframe for prediction
    future_start = datetime(2024, 4, 17)
    future_end = future_start + timedelta(days=365)
    forecast_next_year = forecast.loc[forecast['ds'] >= future_start]

    # Identify high-demand months
    high_demand_months = forecast_next_year.groupby(forecast_next_year['ds'].dt.month)['yhat'].mean().sort_values(ascending=False).head(3).index

    # Filter forecast DataFrame for relevant columns
    dfp = forecast[['ds', 'yhat']]

    # Combine real and predicted data for plotting
    product_df['source'] = 'Real'
    dfp['source'] = 'Predicted'
    combined_df = pd.concat([product_df.rename(columns={'date': 'ds', 'sales': 'y'}), dfp.rename(columns={'yhat': 'y'})])

    # Add day of the week column to predicted sales data
    dfp['day_of_week'] = dfp['ds'].dt.day_name()

    # Calculate mean predicted sales for each day of the week
    avg_trend_by_day = dfp.groupby('day_of_week')['yhat'].mean().reset_index()

    # Convert DataFrame to dictionary and format dates to ISO format
    product_data['product_df'] = product_df.to_dict(orient='records')
    product_data['dfp'] = dfp.to_dict(orient='records')

    for record in product_data['product_df']:
        record['ds'] = record['ds'].isoformat()

    for record in product_data['dfp']:
        record['ds'] = record['ds'].isoformat()

    return JsonResponse(product_data)
