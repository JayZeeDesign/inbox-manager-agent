from dotenv import find_dotenv, load_dotenv

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from custom_tools import CreateEmailDraftTool, GenerateEmailResponseTool, ReplyEmailTool, EscalateTool, ProspectResearchTool, CategoriseEmailTool

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

system_message = SystemMessage(
    content="""
    You are an email inbox assistant of a Realtor named Jacob Ferrari, 
    who deals with buy, sells, rental, and adminstrative emails, 
    Your goal is to handle all the incoming emails by categorising them based on 
    guideline and decide on next steps
    """
)

tools = [
    CategoriseEmailTool(),
    ProspectResearchTool(),
    EscalateTool(),
    ReplyEmailTool(),
    CreateEmailDraftTool(),
    GenerateEmailResponseTool(),
]

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
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


test_email = """
Hi Jacob,

 

I am glad my documents look great. Further to my e-mail, I will be living on my own and I have no pets. Sorry, I just saw your e-mail this morning. What dates and or times are you free this week for a viewing?

 

Thank you so much!

 

Best regards,

 

John (Jay) Newport

Vice President

LMClark-Logo-Blue-50

L.M. Clark Customs Broker Ltd.

1804 Alstep Drive Suite 200

Mississauga, On

L5S 1W1

Direct line: 289-548-5085

Fax:905-673-7345

www.lmclark.com
"""

agent({"input": test_email})
