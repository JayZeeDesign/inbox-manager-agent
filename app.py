import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
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
    guidelines and decide on next steps.
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
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

def serialize(obj):
    """Recursively serialize objects to a dictionary."""
    if isinstance(obj, (str, int, float, bool)):
        return obj  # Simple data types
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    else:
        return str(obj)  # Fallback for unhandled types

@app.route('/process-email', methods=['POST'])
def process_email():
    data = request.get_json()
    email_content = data.get('emailBody')

    if not email_content:
        return jsonify({"error": "No email content provided"}), 400

    try:
        response = agent({"input": email_content})
        response_data = serialize(response)
        serialized_json = json.dumps(response_data)  # Serialize to JSON string
        return jsonify({"message": "Email processed successfully", "response": json.loads(serialized_json)}), 200
    except TypeError as te:
        app.logger.error(f'Serialization error: {str(te)}')
        return jsonify({"error": str(te)}), 500
    except Exception as e:
        app.logger.error(f'Error processing email: {str(e)}')
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
