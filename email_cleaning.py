import os
from dotenv import find_dotenv, load_dotenv
import openai
import json, csv

load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")

def parse_email(email_thread):

    system_prompt = """
    You are an expert of convert raw email thread into original message / reply pairs. 
    You are given a raw email thread that Jason reply to others, your goal is to convert it into original message / reply pairs. 
    - orignal_message: the last message sent to Jason, if it is a long email thread, only take the last message
    - jason_reply: Jason's reply to the original message

    if there is only one message in the thread, that should be jason_reply

    The exported format should look something like 
    {
        "original_message": "xxxx",
        "jason_reply": "xxxx"
    }
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email_thread}
        ]
    )

    return response["choices"][0]["message"]["content"]

def process_csv(input_csv_path, output_csv_path):
    with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        processed_data = []
        
        for row in csv_reader:
            text = row['Body']  # Get the text from the 'body' column
            json_string = parse_email(text)
            print(json_string)
            json_data = json.loads(json_string)  # Convert JSON string to dictionary
            original_message = json_data.get('original_message', '')
            jason_reply = json_data.get('jason_reply', '')
            # Append original row data and new columns to processed_data
            processed_data.append([original_message, jason_reply])
    
    # Write processed data to a new CSV file
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['original_message', 'jason_reply'])
        # Write data rows
        csv_writer.writerows(processed_data)

# Paths to your input and output CSV files
input_csv_path = 'past_email_final_mboxt.csv'
output_csv_path = 'email_pairs.csv'

# Call the function to process the CSV file
process_csv(input_csv_path, output_csv_path)
