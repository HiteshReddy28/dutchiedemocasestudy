import subprocess
import time
import requests
from pathlib import Path

def test_streamlit_app():
    """Test if the Streamlit app starts and responds."""
    try:
        # Start the app in the background
        process = subprocess.Popen(
            ['streamlit', 'run', 'app.py'],
            cwd='.',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a bit for the app to start
        time.sleep(5)

        # Try to access the app
        response = requests.get('http://localhost:8503', timeout=10)

        if response.status_code == 200:
            print("Streamlit app is running and accessible.")
        else:
            print(f"Streamlit app returned status code: {response.status_code}")

        # Terminate the process
        process.terminate()
        process.wait()

    except Exception as e:
        print(f"Error testing Streamlit app: {e}")

if __name__ == "__main__":
    test_streamlit_app()
