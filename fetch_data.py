"""
Script to fetch data from Dutchie API and load it into the warehouse.

This script fetches sales and line items data for all configured locations
from the Dutchie API and loads it into our DuckDB warehouse using the ETL
module. API keys should be configured in a .env file.
"""

from etl import upsert_from_files
from api import fetch_dutchie_exports
from dotenv import load_dotenv
import os

def check_environment():
    """Verify that required environment variables are set."""
    load_dotenv()  # Make sure .env is loaded
    
    from api_constants import LOCATION_API_KEY_FORMAT, INTEGRATOR_KEY_ENV
    
    # Check integrator key
    if not os.getenv(INTEGRATOR_KEY_ENV):
        raise ValueError(f"Missing required environment variable: {INTEGRATOR_KEY_ENV}")
    
    # Check location keys
    locations = ["COLUMBUS", "CINCINNATI"]
    missing = []
    for loc in locations:
        key_name = LOCATION_API_KEY_FORMAT.format(loc)
        if not os.getenv(key_name):
            missing.append(key_name)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
    print("Environment variables loaded successfully.")

def main():
    try:
        # Verify environment setup
        check_environment()
        
        # Fetch Columbus data
        sales_col = fetch_dutchie_exports("COLUMBUS", endpoint="sales")
        items_col = fetch_dutchie_exports("COLUMBUS", endpoint="line_items")

        # Fetch Cincinnati data  
        sales_cin = fetch_dutchie_exports("CINCINNATI", endpoint="sales")
        items_cin = fetch_dutchie_exports("CINCINNATI", endpoint="line_items")

        # Run ETL
        upsert_from_files(
            sales_files=[sales_col, sales_cin],
            item_files=[items_col, items_cin]
        )
        print("Data fetched and loaded successfully!")

    except Exception as e:
        print(f"Error fetching/loading data: {e}")

if __name__ == "__main__":
    main()