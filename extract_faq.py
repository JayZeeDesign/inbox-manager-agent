import csv
import json
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from tqdm import tqdm  # Import tqdm for the progress bar

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4-1106-preview"
)

# Function to load data from CSV
def load_csv(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data_list.append(row)
    return data_list

# Function to extract FAQs
def extract_faq(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False)

    texts = text_splitter.split_text(text_data)
    docs = text_splitter.create_documents(texts)

    map_prompt = """
    PAST EMAILS:
    {text}
    ----
    You are a smart AI assistant, above are some past emails from Jacob Ferrari (a real estate agent), 
    your goal is to learn & extract common FAQ about Jacob Ferrari 
    (include both question & answer, return results in JSON):
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    The following is a set of FAQs about Jacob Ferrari (a real estate agent):
    {text}
    Take these and distill them into a final, consolidated array of FAQs, 
    include both question & answer (in JSON format). 
    
    Array of FAQs:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True
    )

    faqs = []
    for doc in tqdm(docs, desc="Extracting FAQs"):
        partial_output = summary_chain.run([doc])
        # Strip markdown formatting before JSON parsing
        formatted_output = partial_output.replace("```json", "").replace("```", "").strip()

        # Parse the JSON output to append FAQs
        try:
            parsed_json = json.loads(formatted_output)
            if isinstance(parsed_json, list):
                faqs.extend(parsed_json)
            else:
                faqs.extend(parsed_json.get("FAQs", []))
            if len(faqs) >= 200:  # Stop if 200 FAQs are reached
                break
        except json.JSONDecodeError:
            print("Failed to parse JSON.")

    return faqs[:200]  # Return only up to 200 FAQs

# Function to save JSON data to CSV
def save_json_to_csv(data, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for faq in data:
            writer.writerow({'question': faq.get('question'), 'answer': faq.get('answer')})

# Main script execution
if __name__ == "__main__":
    past_emails = load_csv("email_pairslong.csv")
    jacobs_replies = [entry["jacob_reply"] for entry in past_emails]
    jacobs_replies_string = json.dumps(jacobs_replies)

    faqs = extract_faq(jacobs_replies_string)
    save_json_to_csv(faqs, "faq.csv")
