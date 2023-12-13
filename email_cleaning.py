import os
import json
import csv
import asyncio
import aiohttp
from dotenv import find_dotenv, load_dotenv

# Setting the event loop policy for Windows to prevent compatibility issues
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables, including the OpenAI API key
load_dotenv(find_dotenv())
api_key = os.environ.get("OPENAI_API_KEY")

# Function to asynchronously parse an email thread using OpenAI API
async def parse_email(session, email_thread):
    # System prompt defining the task for the OpenAI model
    system_prompt = """
    [Your system prompt here]
    """

    # Preparing the request body with the model, messages, and response format
    json_body = {
        "model": "gpt-4-1106-preview",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email_thread}
        ],
        "response_format": {"type": "json_object"}
    }

    # Headers for the API request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    # Making an asynchronous POST request to the OpenAI API
    async with session.post('https://api.openai.com/v1/chat/completions', json=json_body, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            return data['choices'][0]['message']['content']
        else:
            print("Error with OpenAI API:", response.status)
            response_text = await response.text()
            print("Response content:", response_text)
            return None

# Function to process a batch of emails
async def process_batch(session, email_threads):
    return await asyncio.gather(*[parse_email(session, email) for email in email_threads])

# Main function to process the CSV file and extract email pairs
async def process_csv(input_csv_path, output_csv_path):
    async with aiohttp.ClientSession() as session:
        with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            processed_data = []
            batch_size = 10  # Batch size for processing emails

            # Reading and processing each email in the CSV file
            for row in csv_reader:
                email_batch.append(row['Body'])
                if len(email_batch) >= batch_size:
                    results = await process_batch(session, email_batch)
                    for email_thread, json_string in zip(email_batch, results):
                        if json_string:
                            try:
                                json_data = json.loads(json_string)
                                original_message = json_data.get('original_message', '')
                                jason_reply = json_data.get('jason_reply', '')
                                processed_data.append([original_message, jason_reply])
                            except json.JSONDecodeError:
                                print(f"Failed to decode JSON from response: {json_string}")
                    email_batch = []

            # Processing remaining emails in the final batch
            if email_batch:
                results = await process_batch(session, email_batch)
                for email_thread, json_string in zip(email_batch, results):
                    if json_string:
                        try:
                            json_data = json.loads(json_string)
                            original_message = json_data.get('original_message', '')
                            jason_reply = json_data.get('jason_reply', '')
                            processed_data.append([original_message, jason_reply])
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON from response: {json_string}")

        # Writing the processed data to a new CSV file
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['original_message', 'jason_reply'])
            csv_writer.writerows(processed_data)

# Input and output file paths (adjust as needed)
input_csv_path = 'input_csv_path'
output_csv_path = 'output_csv_path'

# Running the main function to process the CSV file
asyncio.run(process_csv(input_csv_path, output_csv_path))
