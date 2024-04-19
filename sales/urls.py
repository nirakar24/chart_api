from django.urls import path
from . import views

urlpatterns = [
    path('forecast_sales/<int:product_id>/', views.forecast_sales, name='forecast_sales'),
    path('forecast_sales/<str:product_ids>/<str:csv_file>/', views.forecast_sales, name='forecast_sales_csv'),
    path('forecast_demand_level/<int:product_id>/', views.forecast_demand_level, name='forecast_demand_level'),
    path('optimal_placement/<int:product_id>/', views.optimal_placement, name="optimal_placement"),
    path('calculate_reorder_alerts/<int:product_id>/', views.calculate_reorder_alerts, name='calculate_reorder'),
    path('sentimental_analysis/<str:keyword>/', views.sentiment_analysis_view, name="sentiment_analysis"),

]
