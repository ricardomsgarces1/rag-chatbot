import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json

def fetch_html(url):
    """
    Fetch the HTML content of the webpage.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The HTML content of the webpage.

    Raises:
        HTTPError: If the HTTP request returns an unsuccessful status code.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def extract_webJson(html_content):
    """
    Extract the 'webJson' variable from the HTML content.

    Args:
        html_content (str): The HTML content of the webpage.

    Returns:
        str or None: The 'webJson' variable as a string, or None if not found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tags = soup.find_all('script')
    
    for script in script_tags:
        if script.string and 'webJson' in script.string:
            return script.string

    return None


def extract_urls_from_webJson(webJson_script, api_key):
    """
    Use an LLM (OpenAI API) to extract URLs from the 'webJson' variable.

    Args:
        webJson_script (str): The 'webJson' JavaScript code as a string.
        api_key (str): OpenAI API key for authentication.

    Returns:
        str: The extracted URLs as a JSON object.
    """
    prompt = (
        "Extract the list of URLs 'url' in the JSON structure with the following name 'webJson'. "
        "Ignore the 'url' values associated with a 'tag' value set to 'h2'. \n\n"
        f"JavaScript code:\n\n{webJson_script}\n\n"
        "Return the URLs as a JSON array of objects with the structure {url, name}. Extract the values directly."
    )

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }
    )

    # Extract the content from the response
    urls = response.choices[0].message.content
    return urls


def save_json_to_file(json_content, filename):
    """
    Save the JSON content to a file.

    Args:
        json_content (str): The JSON content as a string.
        filename (str): The name of the file to save the content.

    Returns:
        None
    """
    try:
        data = json.loads(json_content)  # Ensure valid JSON

        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(f"JSON content successfully written to {filename}.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


def main():
    """
    Main function to orchestrate the script's workflow.

    Steps:
        1. Fetch HTML from the provided URL.
        2. Extract the 'webJson' variable from the HTML.
        3. Use OpenAI to extract URLs from 'webJson'.
        4. Save the extracted URLs to a JSON file.
    """
    # URL of the webpage to process
    url = input("Enter the URL of the webpage to process: ")
    
    # Fetch HTML content
    html_content = fetch_html(url)
    
    # Extract the 'webJson' variable
    webJson_script = extract_webJson(html_content)

    if webJson_script:
        # OpenAI API key (replace with your actual key)
        api_key=os.environ.get("OPENAI_API_KEY")
        
        # Extract URLs using OpenAI API
        urls = extract_urls_from_webJson(webJson_script, api_key)
        print("Extracted URLs:", urls)
        
        # Save the extracted URLs to a JSON file
        filename = input("Enter the filename to save the JSON content (e.g., output.json): ")
        save_json_to_file(urls, "urls/" + filename)
    else:
        print("The 'webJson' variable was not found in the HTML content.")


if __name__ == "__main__":
    main()
