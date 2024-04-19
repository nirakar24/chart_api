from django.urls import path
from . import views

urlpatterns = [
    path('forecast_sales/<int:product_id>/', views.forecast_sales, name='forecast_sales'),
    path('forecast_sales/<str:product_ids>/<str:csv_file>/', views.forecast_sales, name='forecast_sales_csv'),

]
