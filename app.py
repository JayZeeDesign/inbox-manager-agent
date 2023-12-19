from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage
from custom_tools import (
    CreateEmailDraftTool, GenerateEmailResponseTool, 
    ReplyEmailTool, EscalateTool, ProspectResearchTool, CategoriseEmailTool
)

app = Flask(__name__)
load_dotenv()

# Initialize your LangChain agent
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

system_message = SystemMessage(
    content="""
    You are an email inbox assistant of a Realtor named Jacob Ferrari, 
    who deals with buy, sells, rental, and administrative emails, 
    Your goal is to handle all the incoming emails by categorizing them based on 
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
    "extra_prompt_messages": [],
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

@app.route('/process-email', methods=['POST'])
def process_email():
    data = request.get_json()
    email_content = data.get('emailBody')

    if not email_content:
        return jsonify({"error": "No email content provided"}), 400

    try:
        # Process the email content with your agent
        response = agent({"input": email_content})
        return jsonify({"message": "Email processed successfully", "response": response}), 200
    except Exception as e:
        # Log the error to your server logs
        app.logger.error(f'Error processing email: {str(e)}')
        # Return the error message in the response for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)