import json
import re

def extract_json_from_string(input_string):
    # Define a regular expression pattern to match a JSON object
    # This pattern will look for curly braces and attempt to match the JSON structure
    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    
    # Search for the JSON pattern in the input string
    match = json_pattern.search(input_string)
    
    if match:
        json_str = match.group(0)
        try:
            # Parse the JSON string into a Python dictionary
            json_dict = json.loads(json_str)
            return json_dict
        except json.JSONDecodeError:
            # If JSON is not properly formatted, raise an error
            raise ValueError("Found JSON object is not properly formatted")
    else:
        # If no JSON object is found, return None or raise an error
        raise ValueError("No JSON object found in the input string")