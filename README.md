# Warehouse Management System API

This project is an API designed to enhance warehouse efficiency by providing forecasts for sales, demand levels, optimal product placement, reorder alerts, and sentiment analysis. The API supports versatile input types, including individual products and CSV file uploads.

## Features

- **Sales Forecasting**: Predicts future sales for individual products or a batch of products via CSV input.
- **Demand Level Forecasting**: Estimates demand levels to help optimize inventory management.
- **Optimal Product Placement**: Suggests the best placement for products in a warehouse to maximize efficiency.
- **Reorder Alerts**: Generates alerts when a product needs to be reordered based on forecasted demand.
- **Sentiment Analysis**: Analyzes customer feedback or product reviews to gauge sentiment and inform business decisions.

## API Endpoints

1. **Sales Forecasting**
   - **POST `/api/forecast/sales`**
   - **Input**: Product details or a CSV file containing historical sales data.
   - **Output**: Sales forecast for the specified period.

2. **Demand Level Forecasting**
   - **POST `/api/forecast/demand`**
   - **Input**: Product details or CSV file.
   - **Output**: Forecasted demand levels.

3. **Optimal Product Placement**
   - **POST `/api/optimize/placement`**
   - **Input**: Product details and warehouse layout.
   - **Output**: Suggested placement strategy for the product.

4. **Reorder Alerts**
   - **POST `/api/alerts/reorder`**
   - **Input**: Product details and current inventory levels.
   - **Output**: Reorder alerts with recommended quantities.

5. **Sentiment Analysis**
   - **POST `/api/analyze/sentiment`**
   - **Input**: Text input such as customer feedback or product reviews.
   - **Output**: Sentiment analysis report (e.g., positive, negative, neutral).

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/sales-demand-forecasting-api.git
   cd sales-demand-forecasting-api
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   - Create a `.env` file in the project root and configure any necessary environment variables.

5. **Run Migrations**
   ```bash
   python manage.py migrate
   ```

6. **Run the Development Server**
   ```bash
   python manage.py runserver
   ```

7. **Access the API**
   - The API will be accessible at `http://localhost:8000`.

## Usage

- Use a tool like Postman or cURL to interact with the API endpoints.
- For sales forecasting, either provide product details directly or upload a CSV file with historical sales data.
- Use the API to receive forecasts, alerts, and analyses to improve decision-making processes.

## Technologies Used

- **Django**: Web framework used for building the API.
- **Pandas**: For handling CSV inputs and data manipulation.
- **Scikit-learn**: For implementing forecasting and machine learning models.
- **NLTK/Spacy**: For performing sentiment analysis.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact Nirakar Jena at [Mail](mailto:jenashubham60@gmail.com).
