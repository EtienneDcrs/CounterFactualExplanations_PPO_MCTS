# test_server.py
import logging
from fastmcp.client.transports import StdioTransport
from fastmcp import Client
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

transport = StdioTransport(
    command="python",
    args=["server.py"],
    cwd="C:\\Users\\etien\\Desktop\\Internship XAI Code\\CounterFactualExplanations_PPO_MCTS"
)
client = Client(transport)

try:
    # Initialize session
    init_response = client.invoke("initialize", {})
    print("Initialize Response:", json.dumps(init_response, indent=2))
    # Test resource
    response = client.invoke("get_resource", {"resource": "counterfactual://dataset/info"})
    print("Dataset Info:", json.dumps(json.loads(response), indent=2))
    # Test tool listing
    tools = client.invoke("list_tools", {})
    print("Available Tools:", json.dumps(tools, indent=2))
    # Test tool call
    tool_response = client.invoke("train_ppo_tool", {"dataset_path": "data/breast_cancer.csv"})
    print("Tool Response:", json.dumps(tool_response, indent=2))
except Exception as e:
    print(f"Error: {e}")
    print("Available Client methods:", dir(client))