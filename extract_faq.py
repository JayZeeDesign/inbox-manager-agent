import csv
import json
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

load_dotenv()
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

def load_csv(file_path):
    # Create a list to hold dictionaries
    data_list = []

    # Open the CSV file and read its content
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # For each row, append it as a dictionary to the list
        for row in csv_reader:
            data_list.append(row)

    return data_list

def extract_faq(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=20,
        length_function = len,
        is_separator_regex=False)

    texts = text_splitter.split_text(text_data)
    docs = text_splitter.create_documents(texts)


    map_prompt = """
    PAST EMAILS:
    {text}
    ----

    You are a smart AI assistant, above is some past emails from AI Jason (an AI youtuber), 
    your goal is to learn & extract common FAQ about AI Jason 
    (include both question & answer, return results in JSON):
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    The following is set of FAQ about AI Jason (an AI youtuber):
    {text}
    Take these and distill it into a final, consolidated array of faq, 
    include both question & answer (in JSON format). 
    
    array of FAQ:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
                                        verbose=True
                                        )

    output = summary_chain.run(docs)
    faqs = json.loads(output)

    return faqs

def save_json_to_csv(data, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        # Get the keys (column names) from the first dictionary in the list
        fieldnames = data[0].keys()
        
        # Create a CSV dict writer object
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write the data rows
        for entry in data:
            writer.writerow(entry)


# Print or save the JSON data
past_emails = load_csv("email_pairs.csv")

# Extracting Jason's replies
jasons_replies = [entry["jason_reply"] for entry in past_emails]
jasons_replies_string = json.dumps(jasons_replies)

faqs = extract_faq(jasons_replies_string)

save_json_to_csv(faqs, "faq.csv")

