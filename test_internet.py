import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Define a Retry object with backoff factor
retries = Retry(total=100, backoff_factor=1, status_forcelist=[502, 503, 504])

# Create an HTTPAdapter with the retry object
adapter = HTTPAdapter(max_retries=retries)

# Create a session
session = requests.Session()

# Mount the adapter to the session
session.mount('http://', adapter)
session.mount('https://', adapter)

# Now you can use the session to make requests
try:
    response = session.get('http://discovery.informatics.uab.edu/PAGER/index.php/browse/gs_by_supertree/Bacterial%20Infections')
    response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
except requests.exceptions.HTTPError as errh:
    print(f"Http Error: {errh}")
except requests.exceptions.ConnectionError as errc:
    print(f"Error Connecting: {errc}")
except requests.exceptions.Timeout as errt:
    print(f"Timeout Error: {errt}")
except requests.exceptions.RequestException as err:
    print(f"OOps: Something Else: {err}")

