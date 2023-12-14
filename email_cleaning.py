import os
import json
import csv
import asyncio
import aiohttp
from dotenv import find_dotenv, load_dotenv

# Setting the event loop policy for Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv(find_dotenv())
api_key = os.environ.get("OPENAI_API_KEY")

async def parse_email(session, email_thread):
    system_prompt = """
    You are an expert in converting raw email threads into original message / reply pairs. 
    You are given a raw email thread that Jason replies to others, your goal is to convert it into original message / reply pairs. 
    - jason_reply: Jason's reply to the original message. It will ALWAYS start off as a greeting 'Hi/Hey/Hello ____ (insert name), the blablabla ....'
    - original_message: any message sent to Jason, that DOES NOT start with a greeting such as 'Hi/Hey/Hello ____ (insert name)'
    
    if there is only one message in the thread, that whole message should be jason_reply

    The exported format should be in JSON format and look like 
    {
        "original_message": "Hope your day is well ... ",
        "jason_reply": "Hi (name)..."
    }
    NEVER start it off like this ```json or end it with ```
    """

    json_body = {
        "model": "gpt-4-1106-preview",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email_thread}
        ],
        "response_format": {"type": "json_object"}
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    async with session.post('https://api.openai.com/v1/chat/completions', json=json_body, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            return data['choices'][0]['message']['content']
        else:
            print("Error with OpenAI API:", response.status)
            response_text = await response.text()
            print("Response content:", response_text)
            return None

async def process_batch(session, email_threads):
    return await asyncio.gather(*[parse_email(session, email) for email in email_threads])

async def process_csv(input_csv_path, output_csv_path):
    async with aiohttp.ClientSession() as session:
        with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            processed_data = []
            batch_size = 10  # Adjust the batch size as needed
            email_batch = []

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

        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['original_message', 'jason_reply'])
            csv_writer.writerows(processed_data)

# Your existing code to set file paths and call process_csv

# Paths to your input and output CSV files
input_csv_path = 'input_csv_path'
output_csv_path = 'output_csv_path'

# Run the asynchronous process
asyncio.run(process_csv(input_csv_path, output_csv_path))
