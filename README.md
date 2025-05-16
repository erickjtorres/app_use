# App Use - AI Control for Mobile Apps

> **Warning:** This project is currently under active development. APIs will change! Currently in experimental stages.

App Use is the easiest way to connect AI agents with mobile applications. It enables AI to control your mobile apps through native debugging bridges, providing a powerful yet simple interface for app automation.

Currently, App Use focuses on Flutter apps via the Dart VM Service, with plans to expand to other frameworks like React Native in the future.

## Overview

App Use allows you to automate interactions with mobile apps, such as:
- Clicking buttons and interactive widgets
- Entering text into fields
- Scrolling and navigating screens
- Building AI-powered agents that can interact with mobile app UIs

Perfect for testing, automation, and AI agent applications.

## Getting Started

### Installation

Install the package using pip:

```bash
pip install -r requirements.txt
```

### Quick Start with LLM Agents

Use AI agents to intelligently interact with your mobile app:

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app_use.agent.service import Agent
from app_use.app.app import App

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

async def main():
    # WebSocket URI for a running Flutter app
    VM_SERVICE_URI = "ws://127.0.0.1:50505/ws"
    
    # Set up your LLM
    llm = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",  # or any other model
        temperature=0
    )
    
    # Create an App instance to connect to the Flutter application
    app = App(vm_service_uri=VM_SERVICE_URI)
    
    # Create an agent to control the app
    agent = Agent(
        task="Explore the app and find all buttons, then click on the Login button if it exists.",
        llm=llm,
        app=app,
        max_steps=10  # Limit the number of steps the agent will take
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
        app.close()

if __name__ == "__main__":
    asyncio.run(main())

### Setting Up a Flutter App for Control

> Currently, App Use supports Flutter apps, with more frameworks coming soon.

1. Run your Flutter app with the VM service enabled:

```bash
flutter run --observe
```

2. Look for the VM service URI in the console output, which looks like:
   `Observatory listening on http://127.0.0.1:50505/xyzabcdef=/`

3. Use the WebSocket version of this URI in your code: 
   `ws://127.0.0.1:50505/ws`

## Customize

### Widget Interaction Methods

App Use provides several methods to interact with UI elements:

- `click_widget_by_unique_id`: Click a widget, with intelligent fallback to parent/child widgets if needed
- `enter_text_with_unique_id`: Enter text into a widget
- `scroll_into_view`: Ensure a widget is visible
- `scroll_up_or_down`: Scroll in a specific direction

Each method tries multiple approaches to ensure successful interaction, including:
- By Flutter key
- By text content
- By widget type
- By ancestor-descendant relationship
- By element key/id
- By text content
- By element type
- By ancestor-descendant relationship

## Features

- **Smart Widget Interaction**: Automatically attempts to interact with parent or child widgets if direct interaction fails
- **Automatic Resource Management**: Proper cleanup of connections and services
- **Context Manager Support**: Use the `with` statement for proper resource handling
- **Intelligent Port Management**: Automatically handles port conflicts
- **Extensive UI Tree Access**: Full access to the application's UI tree
- **Flexible Element Finding**: Find UI elements by key, text, type, or hierarchical position

### Extending App Use

You can extend App Use by:
- Creating custom Agent implementations
- Adding new widget interaction patterns
- Adding support for additional app frameworks
- Building domain-specific UI automation tools

## Community & Support

Contributions are welcome! Please feel free to submit a Pull Request.

App Use is actively maintained and designed to make mobile app control as simple and reliable as possible.

        