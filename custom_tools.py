import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")


# CATEGORISE EMAIL
def check_consult_inquiry(lates_reply: str):
    prompt = f"""
    EMAIL: {lates_reply}
    ---

    Above is an email about a property inquiry; Your goal is to identify if all necessary information is provided by the client for a consultation on buying, selling, or renting properties:
    1. The type of property the client is interested in.
    2. The location of interest.
    3. The client's budget.
    4. The client's timeline or urgency.
    5. Any specific requirements or preferences for the property.

    If all the information above is collected, return YES, otherwise, return NO; (Return ONLY YES or NO)

    ANSWER: 
    """

    all_needs_collected_result = client.chat.completions.create(model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": prompt}
    ])

    all_needs_collected = all_needs_collected_result.choices[0].message.content

    return all_needs_collected


def categorise_email(lates_reply: str):
    categorise_prompt = f"""
    EMAIL: {lates_reply}
    ---

    Your goal is to categorise the email based on categories below:

    1. COLLABORATION: These are emails where companies or individuals are reaching out to propose a collaboration with Jacob Ferrari. They often include details about their product or service and how they envision the partnership.

    2. CONSULTING: These emails involve individuals reaching out to the real estate agent with specific requests regarding property transactions. This could range from seeking advice on purchasing a new home or selling a home.

    3. QUESTIONS: These emails involve individuals reaching out to Jacob Ferrari with specific questions or inquiries. This could be about real estate, his knowledge on a specific topic, or his thoughts on the market.

    4. NON_REPLY: These are auto emails that don't need any response or involve companies or individuals reaching out to Jacob Ferrari to offer their services. This could be a marketing agency offering to help him find sponsorship opportunities or a company offering a specific tool or service they think he might find useful.

    5. OTHER: These are emails that don't fit into any of the above categories.

    CATEGORY (Return ONLY the category name in capital):
    """

    category_result = client.chat.completions.create(model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": categorise_prompt}
    ])

    category = category_result.choices[0].message.content

    if category == "CONSULTING":
        all_needs_collected = check_consult_inquiry(lates_reply)
        if all_needs_collected == "YES":
            return {
                "Step 1": """Forward the email to jferrarishaikh@gmail.com, with summary of 1. The type of property the client is interested in. 2. The location of interest. 3. The client's budget. 4. The client's timeline or urgency. 5. Any specific requirements or preferences for the property."""
            }
        else:
            return {
                "Step 1": "Generate email response to the prospect to collect further info based on guidelines",
                "Step 2": "Send generated email response to prospect",
            }
    else:
        if category == "COLLABORATION":
            return {
                "Step 1": "Research about the prospect & company",
                "Step 2": "Forward the email to jferrarishaikh@gmail.com, with the research results included"
            }
        else:
            if category == "NON_REPLY":
                return f"This email has already been taken care of or replied before, nothing needs to be done now"
            else:
                return {
                    "Step 1": "Generate email response based on guidelines",
                    "Step 2": "Create email draft with the generated response"
                }


class CategoriseEmailInput(BaseModel):
    lates_reply: str = Field(description="Latest reply from the prospect ")


class CategoriseEmailTool(BaseTool):
    name = "categorise_email"
    description = "use this to categorise email to decide what to do next"
    args_schema: Type[BaseModel] = CategoriseEmailInput

    def _run(self, lates_reply: str):
        return categorise_email(lates_reply)

    def _arun(self, url: str):
        raise NotImplementedError(
            "get_stock_performance does not support async")


# WRITE EMAIL
def generate_email_response(email_thread: str, category: str):
    # URL endpoint
    url = "https://api-bcbe5a.stack.tryrelevance.com/latest/studios/e97a281a-eea9-4846-bdb8-00b817b51a1c/trigger_limited"

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Payload (data)
    data = {
        "params": {
            "client_email": email_thread,
            "goal": "write email response" if category != "CONSULTING FOLLOW UP" else "for each consulting email, we need to collect 1. The type of property the client is interested in. 2. The location of interest. 3. The client's budget. 4. The client's timeline or urgency. 5. Any specific requirements or preferences for the property; Try to collect those info from them",
        },
        "project": "1ff4f8cf216a-4967-adef-1bfe8d8fb84e"
    }

    # Send POST request
    response = requests.post(url, headers=headers, json=data)

    return response.text


class GenerateEmailResponseInput(BaseModel):
    """Inputs for scrape_website"""
    email_thread: str = Field(description="The original full email thread")
    category: str = Field(
        description='category of email, can ONLY be "CONSULTING FOLLOW UP" or "OTHER" ')


class GenerateEmailResponseTool(BaseTool):
    name = "generate_email_response"
    description = "use this to generate the email response based on specific guidelines, voice & tone & knowledge for Jacob Ferrari"
    args_schema: Type[BaseModel] = GenerateEmailResponseInput

    def _run(self, email_thread: str, category: str):
        return generate_email_response(email_thread, category)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")


# RESEARCH AGENT

def summary(objective, content):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        "https://chrome.browserless.io/content?token=xxxxxxxxxxxxxxxxxxxxxxxxxxx", headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        return f"HTTP request failed with status code {response.status_code}"


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError(
            "get_stock_performance does not support async")


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': 'xxxxxxxxxxxxxxxxxxxxx',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def prospect_research(email_or_name: str, company: str):
    tools = [
        Tool(
            name="Search",
            func=search,
            description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
        ),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
                you do not make things up, you will try as hard as possible to gather facts & data to back up the research
                
                Please make sure you complete the objective above with the following rules:
                1/ You should do enough research to gather as much information as possible about the objective
                2/ If there are url of relevant links & articles, you will scrape it to gather more information
                3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
                4/ You should not make things up, you should only write facts & data that you have gathered
                5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
                6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

    message = f"Research about {company} and {email_or_name}; What does the company do, and who the person is"

    result = agent({"input": message})

    return result


class ProspectResearchInput(BaseModel):
    """Inputs for scrape_website"""
    email_or_name: str = Field(
        description="The original email address or name of prospect")
    company: str = Field(description="The company name of prospect")


class ProspectResearchTool(BaseTool):
    name = "prospect_research"
    description = "useful when you need to research about a prospect, passing both email and company to the function, return the summary of its company as well as the prospect"
    args_schema: Type[BaseModel] = ProspectResearchInput

    def _run(self, email_or_name: str, company: str):
        return prospect_research(email_or_name, company)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")


# ESCALATE

def escalate(original_email_address: str, message: str, additional_context: str):
    # URL to send the POST request to
    url = 'https://hook.us1.make.com/faevm0ijp4yvdmxsnx1yod7q0zqwb8ag'

    # Data to send in the POST request
    data = {
        "prospect email": original_email_address,
        "prospect message": message,
        "additional context": additional_context
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        return ('This email has been escalated to Jacob, he will take care of it from here, nothing needs to be done now')
    else:
        return ('Failed to send POST request:', response.status_code)


class EscalateInput(BaseModel):
    """Inputs for scrape_website"""
    message: str = Field(
        description="The original email thread & message that was received, cc the original copy for escalation")
    original_email_address: str = Field(
        description="The email address that sent the message/email")
    additional_context: str = Field(
        description="additional context about the prospect, can be the company/prospct background research OR the consulting request details like use case, budget, etc.")


class EscalateTool(BaseTool):
    name = "escalate_to_jacob"
    description = "useful when you need to escalate the case to jacob or others, passing both message and original_email_address to the function"
    args_schema: Type[BaseModel] = EscalateInput

    def _run(self, original_email_address: str, message: str, additional_context: str):
        return escalate(original_email_address, message, additional_context)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")


# REPLY EMAIL
def reply_email(message: str, email_address: str, subject: str):
    # URL to send the POST request to
    url = 'https://hook.us1.make.com/kj3b383jpd7z9inhm52bgfnuekjofmc2'

    # Data to send in the POST request
    data = {
        "Email": email_address,
        "Subject": subject,
        "Reply": message
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        return ('Email reply has been created successfully')
    else:
        return ('Failed to send POST request:', response.status_code)



class ReplyEmailInput(BaseModel):
    """Inputs for scrape_website"""
    message: str = Field(
        description="The generated response message to be sent to the email address")
    email_address: str = Field(
        description="Destination email address to send email to")
    subject: str = Field(description="subject of the email")


class ReplyEmailTool(BaseTool):
    name = "reply_email"
    description = "use this to send emails"
    args_schema: Type[BaseModel] = ReplyEmailInput

    def _run(self, message: str, email_address: str, subject: str):
        return reply_email(message, email_address, subject)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")


# CREATE EMAIL DRAFT
def create_email_draft(prospect_email_address: str, subject: str, generated_reply: str):
    # URL to send the POST request to
    url = 'https://hooks.zapier.com/hooks/catch/15616669/38ikw12/'

    # Data to send in the POST request
    data = {
        "email": prospect_email_address,
        "subject": subject,
        "reply": generated_reply
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        return ('Email draft has been created successfully')
    else:
        return ('Failed to send POST request:', response.status_code)


class CreateEmailDraftInput(BaseModel):
    """Inputs for scrape_website"""
    prospect_email_address: str = Field(
        description="The prospect's email address")
    subject: str = Field(description="The original email subject")
    generated_reply: str = Field(
        description="Generated email reply to prospect")


class CreateEmailDraftTool(BaseTool):
    name = "create_email_draft"
    description = "use this to create email draft for jacob to review & send"
    args_schema: Type[BaseModel] = CreateEmailDraftInput

    def _run(self, prospect_email_address: str, subject: str, generated_reply: str):
        return create_email_draft(prospect_email_address, subject, generated_reply)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")
