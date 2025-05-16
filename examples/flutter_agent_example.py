#!/usr/bin/env python3
"""
Flutter App Control Example using app_use Agent with x.ai (Grok).

This example shows how to use the Agent to control a Flutter app using
the dart-vm-client package with x.ai as the LLM. Make sure you have 
a Flutter app running with the VM service enabled.

Usage:
    python flutter_agent_example.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app_use.agent.service import Agent
from app_use.app.app import App


# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

async def main():
    # WebSocket URI for a running Flutter app
    # This is typically output in the console when running a Flutter app with --observatory
    VM_SERVICE_URI = "ws://127.0.0.1:64824/vF1gcOo0Ce0=/ws"
    api_key = os.getenv("GROK_API_KEY")
    
    # Set up x.ai (Grok) via OpenAI compatible API
    llm = ChatOpenAI(
        api_key=SecretStr(api_key),
        base_url="https://api.x.ai/v1",
        model='grok-3-beta'
    )
    
    # Create an App instance to connect to the Flutter application
    app = App(vm_service_uri=VM_SERVICE_URI)
    
    # Create an agent to control the app
    agent = Agent(
        task="Can you make a trade for me?", 
        llm=llm,
        app=app,
    )
    
    try:
        # Run the agent
        history = await agent.run()
        
        # Print results
        if history.is_successful():
            print("✅ Agent successfully completed the task!")
        else:
            print("⚠️ Agent couldn't complete the task.")
        
        # Show the final result/message
        print(f"Final result: {history.final_result()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Make sure to close resources
        await agent.close()
        # Close the App to clean up resources
        app.close()

if __name__ == "__main__":
    asyncio.run(main()) 