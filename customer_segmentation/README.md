# Customer Segmentation Flask API

## Project Description
Developed a Customer Segmentation API using Python, Flask, and K-Means clustering. The API accepts customer information in JSON format, predicts the customer's cluster, and returns the result as a JSON response.

## Technologies Used
- Python
- Flask
- Scikit-learn
- NumPy
- Joblib
- K-Means Clustering

## Features
- Trained K-Means model on customer data.
- Saved the trained model using Joblib.
- Created a Flask API to accept customer features.
- Predicted customer cluster labels.
- Returned predictions in JSON format.

## Input Features
- Annual Income
- Spending Score

## API Endpoint
**POST** `/predict`

### Example Request
```json
{
  "annual_income_k": 50,
  "spending_score": 60
}
```

### Example Response
```json
{
  "cluster_label": 2,
  "message": "Customer belongs to cluster 2"
}
```

## How to Run
1. Install required libraries:
   `pip install flask scikit-learn numpy joblib`
2. Run the Python file:
   `python customer_segmentation.py`
3. Send customer data to the `/predict` endpoint using Postman.

## Conclusion
This project demonstrates customer segmentation using K-Means clustering and Flask API integration to predict customer cluster labels from input data.