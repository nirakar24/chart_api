import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Sales
import json
import requests
# views.py

import asyncio
from django.http import JsonResponse
import pandas as pd
from twscrape import API, gather
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Function to perform sentiment analysis on tweets
async def perform_sentiment_analysis(search_keyword):
    # Initialize Twitter API
    api = API()

    # Add Twitter account credentials and login
    await api.pool.add_account("@its_Jainisha", "Jainisha7506", "jainishatwitter@gmail.com", "jainisha7506")
    await api.pool.login_all()

    # Search for tweets with the provided search keyword
    tweets = await gather(api.search(search_keyword, limit=200, kv={"product": "Top"}))

    # Initialize empty list to store tweet data
    tweet_data = []

    # Loop through retrieved tweets and extract relevant information
    for tweet in tweets:
        tweet_data.append({
            "Date": tweet.date,
            "Tweet": tweet.rawContent,
            "Likes": tweet.likeCount,
            "Retweets": tweet.retweetCount
        })

    # Create a DataFrame from the collected tweet data
    df = pd.DataFrame(tweet_data)

    # Load pre-trained sentiment analysis model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # Preprocess tweet for sentiment analysis
    def preprocess_tweet(tweet):
        processed_tweet = []
        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            processed_tweet.append(word)
        return " ".join(processed_tweet)

    # Function to perform sentiment analysis on a tweet
    def analyze_sentiment(tweet):
        tweet_proc = preprocess_tweet(tweet)
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
        output = model(**encoded_tweet)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        sentiment_score = scores[2]  # Positive class score
        if sentiment_score > 0.3:
            sentiment_category = 'Positive'
        elif 0.05 <= sentiment_score <= 0.3:
            sentiment_category = 'Negative'
        else:
            sentiment_category = 'Neutral'
        return sentiment_score, sentiment_category

    # Apply sentiment analysis function to each tweet and create new columns for sentiment score and sentiment category
    df[['Sentiment Score', 'Sentiment']] = df['Tweet'].apply(lambda x: pd.Series(analyze_sentiment(x)))

    # Extract date and sentiment columns for further analysis
    pos_neg_date = df[['Date', 'Sentiment']]
    return pos_neg_date

# Django View Function
def sentiment_analysis_view(request, keyword):
    # Run the sentiment analysis function with the provided keyword
    pos_neg_date = asyncio.run(perform_sentiment_analysis(f"#{keyword}"))
    
    # Convert DataFrame to JSON response
    response = pos_neg_date.to_json(orient='records')
    return JsonResponse(response, safe=False)



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

@csrf_exempt
def forecast_demand_level(product_id):
    # Fetch data from the Sales table for the specified product_id
    sales_data = Sales.objects.filter(product=product_id).values('date', 'sales')
    df = pd.DataFrame(list(sales_data))

    # Prepare the data for Prophet
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'sales': 'y'})

    # Instantiate Prophet model
    model = Prophet()

    # Fit the model
    model.fit(df)

    # Define future date range
    future_start = datetime(2024, 4, 17)
    future_end = future_start + timedelta(days=365)
    future_df = model.make_future_dataframe(periods=(future_end - future_start).days)

    # Forecast
    forecast = model.predict(future_df)

    # Group by month and calculate mean forecasted sales for each month
    monthly_sales = forecast.loc[forecast['ds'] >= future_start].groupby(forecast['ds'].dt.month)['yhat'].mean()

    # Define thresholds for high, medium, and low demand
    high_threshold = monthly_sales.quantile(0.75)  # Top 25% of months
    low_threshold = monthly_sales.quantile(0.25)   # Bottom 25% of months

    # Identify high, medium, and low-demand months
    high_demand_months = monthly_sales[monthly_sales >= high_threshold]
    low_demand_months = monthly_sales[monthly_sales <= low_threshold]
    medium_demand_months = monthly_sales[(monthly_sales < high_threshold) & (monthly_sales > low_threshold)]

    # Determine the demand level for the current month
    current_month = datetime.now().month
    if current_month in high_demand_months.index:
        demand_level = "high"
    elif current_month in medium_demand_months.index:
        demand_level = "medium"
    elif current_month in low_demand_months.index:
        demand_level = "low"
    else:
        demand_level = "unknown"  # Just in case, if the month doesn't match any demand level

    # Construct a JSON response with the demand level
    response_data = {
        'product_id': product_id,
        'current_month_demand_level': demand_level
    }

    return JsonResponse(response_data)

def optimal_placment_intermediate(product_id):
    
    preferred_locations = {
      3 : (1,1,1),
      5 : (2,3,2),
      1 : (3, 1, 1),
}
    response = forecast_demand_level(product_id)
    forecast_data = json.loads(response.content)
    forecast_data = forecast_data['current_month_demand_level']
    
    def is_location_occupied(row, col, level):
    # Call the provided API to check if the location is occupied
    # Replace this with the actual API call
        return False
    
    def iterate_locations(level_range, row_range, col_range):
      for level in level_range:
          for row in row_range:
              for col in col_range:
                  if not is_location_occupied(row, col, level):
                      return row, col, level
      return None
    
    def search_high_demand_location():
    # Search in rows 1-6, levels 1-3
        location = iterate_locations(range(1, 4), range(1, 7), range(1, 16))
        if location:
            return location

        # If not found in rows 1-6, levels 1-3, search in the rest of the rows, levels 1-3
        location = iterate_locations(range(1, 4), range(7, 17), range(1, 16))
        if location:
            return location

        # If still not found, search in any level
        location = iterate_locations(range(1, 6), range(1, 17), range(1, 16))
        return location
    
    def find_first_available_location():
      for level in range(1, 6):
          for row in range(1, 17):
              for col in range(1, 16):
                  if not is_location_occupied(row, col, level):
                      return row, col, level
      return None
  
    def search_any_available_location():
      return iterate_locations(range(1, 6), range(1, 17), range(1, 16))
  
    def search_low_demand_location():
      # Search in rows 7-last, any level
      location = iterate_locations(range(1, 6), range(7, 17), range(1, 16))
      if location:
          return location

      # If not found in rows 7-last, search in rows 1-6, levels 3-5
      location = iterate_locations(range(3, 6), range(1, 7), range(1, 16))
      return location
  
    def search_regular_demand_location():
        # Search in rows 1-6, levels 4-5
        location = iterate_locations(range(4, 6), range(1, 7), range(1, 16))
        if location:
            return location

        # If not found in rows 1-6, levels 4-5, search in rows 1-6, any level
        location = iterate_locations(range(1, 6), range(1, 7), range(1, 16))
        if location:
            return location

        # If still not found, search in the rest of the rows, any level
        location = iterate_locations(range(1, 6), range(7, 17), range(1, 16))
        return location
    
    if product_id in preferred_locations:
      row, col, level = preferred_locations[product_id]
      if not is_location_occupied(row, col, level):
          return row, col, level
      
    if forecast_data == "high":
          location = search_high_demand_location()
          if location:
              return location

    elif forecast_data == "regular":
          location = search_regular_demand_location()
          if location:
              return location

    elif forecast_data == "low":
          location = search_low_demand_location()
          if location:
              return location
          
    location = search_any_available_location()
    if location:
        return location

    location = find_first_available_location()
    return location

def optimal_placement(request, product_id):
    result = optimal_placment_intermediate(product_id)
    
    result_json = {
        'row': result[0],
        'column': result[1],
        'level': result[2]
    }
    
    return JsonResponse(result_json)

@csrf_exempt
def calculate_reorder_alerts(request, product_id):
    # Define current inventory levels for each product (example values)
    inventory_levels = {
        '1': 233,  # Replace 'product_1', 'product_2', etc. with actual product identifiers
        '2': 500,
        '3': 350,
        '4': 500,
        '5': 50
    }

    # Make a request to the forecast_sales API endpoint
    forecast_api_url = f"http://13.126.174.18/forecast_sales/{product_id}/"
    response = requests.get(forecast_api_url)
    # print(type(response.content))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        predicted_data = response.json()

        predicted_data = pd.DataFrame(predicted_data['dfp'])

        # forecast_data = predicted_data.get('data', [])
        

        # predicted_data = pd.DataFrame(forecast_data)
        # print(type(predicted_data))
        # print(type(predicted_data), 'hihi')

        # Specify lead time (number of days)
        lead_time_days = 4  # Change this value to the desired lead time

        # Define a function to calculate reorder alerts and show sales data
        def calculate_reorder_alerts_and_show_sales(inventory_levels, predicted_data, lead_time_days):
            reorder_alerts = []

            # Get the current date
            current_date = datetime.today().date()
            print(predicted_data['ds'])

            # Iterate through each product in the inventory levels
            for product_id, current_inventory in inventory_levels.items():
                # Filter the predicted data for the specified lead time starting from the current date
                predicted_data['ds'] = pd.to_datetime(predicted_data['ds'])
                lead_time_predicted_data = predicted_data[
                    (predicted_data['ds'].dt.date >= current_date) &
                    (predicted_data['ds'].dt.date < current_date + timedelta(days=lead_time_days))
                ]

                # Calculate the average daily demand over the lead time period
                avg_daily_demand = lead_time_predicted_data['yhat'].mean()

                # Calculate remaining inventory after subtracting the average daily demand for each day of lead time
                remaining_inventory = current_inventory - (avg_daily_demand * lead_time_days)

                # Check if remaining inventory is sufficient
                if remaining_inventory < 0:
                    reorder_alerts.append(
                        f"Reorder needed for product {product_id}! Inventory will be exhausted in {lead_time_days} days. "
                        f"Order now to avoid stockouts. Average daily demand: {avg_daily_demand:.2f}"
                    )
                else:
                    reorder_alerts.append(
                        f"Inventory level for product {product_id} is sufficient for the next {lead_time_days} days. "
                        f"Average daily demand: {avg_daily_demand:.2f}"
                    )

                # Display the predicted sales data for the lead time period
                print(f"\nProduct {product_id}:")
                print(f"Average daily demand for the next {lead_time_days} days: {avg_daily_demand:.2f}")
                print("Sales data for the next few days:")
                print(lead_time_predicted_data[['ds', 'yhat']])

            return reorder_alerts

        # Calculate reorder alerts and display sales data for the specified lead time
        reorder_alerts = calculate_reorder_alerts_and_show_sales(inventory_levels, predicted_data, lead_time_days)

        # Construct a JSON response with the reorder alerts
        response_data = {'reorder_alerts': reorder_alerts}

        return JsonResponse(response_data)
    else:
        # If the request to the forecast_sales API fails, return an error response
        error_response = {'error': 'Failed to retrieve forecast data'}
        return JsonResponse(error_response, status=response.status_code)
