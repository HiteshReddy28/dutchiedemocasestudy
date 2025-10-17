"""
API client for fetching data from Dutchie POS API.

This module provides functions to fetch sales and line items data from
the Dutchie API for specific locations, using appropriate authentication
and returning paths to temporary files containing the responses.
"""

import os
import base64
import requests
import tempfile
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from api_constants import LOCATION_API_KEY_FORMAT, INTEGRATOR_KEY_ENV

# Load environment variables from .env file
load_dotenv()

def get_auth_header(location: str) -> Dict[str, str]:
    """
    Create HTTP Basic Auth header using location API key as username
    and integrator key as password.
    """
    location_key_env = LOCATION_API_KEY_FORMAT.format(location.upper())
    location_key = os.getenv(location_key_env)
    integrator_key = os.getenv(INTEGRATOR_KEY_ENV)
    
    if not location_key:
        raise ValueError(f"Location API key not set in environment: {location_key_env}")
    if not integrator_key:
        raise ValueError(f"Integrator key not set in environment: {INTEGRATOR_KEY_ENV}")
    
    # Create base64 encoded auth string
    auth_str = f"{location_key}:{integrator_key}"
    auth_bytes = base64.b64encode(auth_str.encode()).decode()
    
    return {"Authorization": f"Basic {auth_bytes}"}

def fetch_dutchie_exports(location: str, endpoint: str = "sales", verify_ssl: bool = False) -> str:
    """
    Fetch POS export from Dutchie API for a given location.
    
    Args:
        location: Store location (e.g. "COLUMBUS", "CINCINNATI")
        endpoint: API endpoint to fetch from ("sales" or "line_items")
        verify_ssl: Whether to verify SSL certificates. Default False for testing.
    
    Returns:
        Path to a temporary file containing the response (CSV for sales,
        JSON for line items)
    
    Raises:
        ValueError: If API key not found in environment
        requests.RequestException: If API call fails
    """
    api_key_env = f"DUTCHIE_API_KEY_{location.upper()}"
    api_key = os.getenv(api_key_env)
    # Map endpoints to their API paths
    endpoint_paths = {
        "sales": "pos/orders",
        "line_items": "pos/line-items"
    }
    
    if endpoint not in endpoint_paths:
        raise ValueError(f"Invalid endpoint. Must be one of: {list(endpoint_paths.keys())}")
    
    # Get authentication headers
    headers = get_auth_header(location)
    headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    
    # Base URL and endpoint path
    url = f"https://api.pos.dutchie.com/v1/{endpoint_paths[endpoint]}"
    
    # Add query parameters for date range (last 30 days by default)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    params = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "format": "csv" if endpoint == "sales" else "json"
    }

    # If SSL verification is causing issues, you can disable it
    # Note: This should only be used in development/testing
    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        response = requests.get(url, headers=headers, params=params, verify=verify_ssl)
        
        # Print diagnostic information
        print(f"Request URL: {response.url}")
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 401:
            raise ValueError(f"Authentication failed for location {location}. Please check your API keys.")
        elif response.status_code == 403:
            raise ValueError(f"Access forbidden. Please verify API permissions for location {location}.")
            
        response.raise_for_status()  # Raises error for other status codes
        
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {str(e)}")
        raise

    # Save to a temp file
    suffix = ".csv" if endpoint == "sales" else ".json"
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"{location}_{endpoint}_")
    fd.write(response.content)
    fd.flush()
    return fd.name