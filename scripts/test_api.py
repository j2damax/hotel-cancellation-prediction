"""
Example client script for testing the Hotel Cancellation Prediction API.
"""

import requests
import json


def test_health(base_url="http://localhost:8000"):
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.json()


def test_single_prediction(base_url="http://localhost:8000"):
    """Test a single prediction."""
    print("Testing /predict endpoint...")
    
    booking_data = {
        "lead_time": 120,
        "arrival_month": 7,
        "stays_weekend_nights": 2,
        "stays_week_nights": 3,
        "adults": 2,
        "children": 1,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "booking_changes": 1,
        "adr": 95.50,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 2
    }
    
    response = requests.post(
        f"{base_url}/predict",
        json=booking_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(booking_data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.json()


def test_batch_prediction(base_url="http://localhost:8000"):
    """Test batch predictions."""
    print("Testing /predict/batch endpoint...")
    
    bookings_data = [
        {
            "lead_time": 120,
            "arrival_month": 7,
            "stays_weekend_nights": 2,
            "stays_week_nights": 3,
            "adults": 2,
            "children": 1,
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "booking_changes": 1,
            "adr": 95.50,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 2
        },
        {
            "lead_time": 300,
            "arrival_month": 12,
            "stays_weekend_nights": 1,
            "stays_week_nights": 1,
            "adults": 1,
            "children": 0,
            "is_repeated_guest": 0,
            "previous_cancellations": 2,
            "booking_changes": 3,
            "adr": 120.00,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0
        },
        {
            "lead_time": 50,
            "arrival_month": 3,
            "stays_weekend_nights": 2,
            "stays_week_nights": 5,
            "adults": 2,
            "children": 2,
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "booking_changes": 0,
            "adr": 150.00,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 3
        }
    ]
    
    response = requests.post(
        f"{base_url}/predict/batch",
        json=bookings_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Number of bookings: {len(bookings_data)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.json()


def main():
    """Run all tests."""
    print("=" * 80)
    print("Hotel Cancellation Prediction API - Client Test")
    print("=" * 80)
    print()
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        health = test_health(base_url)
        
        if not health.get("model_loaded"):
            print("⚠ Warning: Model not loaded!")
            print("Please run 'python scripts/train.py' first to train the models.\n")
            return
        
        # Test single prediction
        prediction = test_single_prediction(base_url)
        
        # Test batch prediction
        batch_predictions = test_batch_prediction(base_url)
        
        print("=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("Please ensure the API is running with: python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
