from __future__ import annotations

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

from app_use.agent.message_manager.views import MessageMetadata, MessageManagerState
from app_use.agent.views import ActionResult, AgentOutput, AgentStepInfo
from app_use.nodes.app_node import NodeState

logger = logging.getLogger(__name__)


# Decorator utility function for timing method executions (similar to time_execution_sync in browser_use)
def time_execution_sync(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.debug(f"{label}: {(end - start) * 1000:.2f}ms")
            return result
        return wrapper
    return decorator


class MessageManagerSettings(BaseModel):
    """Settings for the message manager"""
    max_input_tokens: int = 128000
    estimated_characters_per_token: int = 3
    image_tokens: int = 800
    include_attributes: list[str] = []
    message_context: str | None = None
    sensitive_data: dict[str, str] | None = None
    available_file_paths: list[str] | None = None


class AppMessagePrompt:
    """Class to generate user messages based on the app state"""
    
    def __init__(
        self,
        node_state: NodeState,
        result: list[ActionResult] | None = None,
        include_attributes: list[str] = [],
        step_info: AgentStepInfo | None = None,
    ):
        self.node_state = node_state
        self.result = result
        self.include_attributes = include_attributes
        self.step_info = step_info
    
    def get_user_message(self, use_vision=False) -> HumanMessage:
        """Convert app state to a human message with optional screenshot"""
        content = []
        
        # Start with text description
        text_content = self._get_text_content()
        
        # If vision is enabled and we have a screenshot, include it
        if use_vision and self.node_state.screenshot:
            content = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.node_state.screenshot}"},
                },
            ]
        else:
            content = text_content
        
        return HumanMessage(content=content)
    
    def _get_text_content(self) -> str:
        """Generate text description of the current app state"""
        text_content = "# Current App State\n\n"
        
        # Include basic app info
        if hasattr(self.node_state, 'app_info'):
            text_content += f"App: {getattr(self.node_state, 'app_info', 'Unknown App')}\n\n"
        
        # Include widget tree information
        text_content += f"Widget count: {len(self.node_state.selector_map)}\n"
        
        # Include widget details that are important for interaction
        interactive_widgets = []
        text_widgets = []
        
        for idx, (node_id, node) in enumerate(self.node_state.selector_map.items()):
            # Skip internal widgets or those without display info (following memory about widget filtering)
            if getattr(node, 'widget_type', '').startswith('_'):
                continue
                
            # Get node highlight index if available
            highlight_idx = getattr(node, 'highlight_index', None)
            if highlight_idx is None:
                continue
                
            # Create widget entry with highlight index
            widget_entry = f"[{highlight_idx}]"
            
            # Add widget type
            if hasattr(node, 'widget_type'):
                widget_entry += f"<{node.widget_type}>"
            
            # Add text content if available
            if hasattr(node, 'text') and node.text:
                widget_entry += f" '{node.text}'"
                text_widgets.append(widget_entry)
            
            # Mark if interactive
            if hasattr(node, 'is_interactive') and node.is_interactive:
                widget_entry += " (interactive)"
                interactive_widgets.append(widget_entry)
        
        # Add interactive widgets section
        if interactive_widgets:
            text_content += "\n## Interactive Widgets\n"
            text_content += "\n".join(interactive_widgets[:25])  # Limit to avoid token explosion
            if len(interactive_widgets) > 25:
                text_content += f"\n... and {len(interactive_widgets) - 25} more interactive widgets"
        
        # Add text widgets section if not already covered
        remaining_text_widgets = [w for w in text_widgets if w not in interactive_widgets]
        if remaining_text_widgets:
            text_content += "\n\n## Text Widgets\n"
            text_content += "\n".join(remaining_text_widgets[:25])  # Limit to avoid token explosion
            if len(remaining_text_widgets) > 25:
                text_content += f"\n... and {len(remaining_text_widgets) - 25} more text widgets"
        
        # Include previous action results if available
        if self.result:
            text_content += "\n\n## Previous Action Results\n"
            for r in self.result:
                if r.extracted_content:
                    text_content += f"Content: {r.extracted_content}\n"
                if r.error:
                    text_content += f"Error: {r.error}\n"
        
        # Include step info if available
        if self.step_info:
            current_step = self.step_info.step_number + 1  # 0-indexed to 1-indexed for display
            max_steps = self.step_info.max_steps
            text_content += f"\n\nStep {current_step}/{max_steps}"
        
        return text_content


class MessageManager:
    """
    Manages the conversation history between the agent and the LLM
    """
    
    def __init__(
        self,
        task: str,
        system_message: SystemMessage,
        settings: MessageManagerSettings = MessageManagerSettings(),
        state: MessageManagerState = MessageManagerState(),
    ):
        self.task = task
        self.settings = settings
        self.state = state
        self.system_prompt = system_message

        # Only initialize messages if state is empty
        if len(self.state.history.messages) == 0:
            self._init_messages()

    def _init_messages(self) -> None:
        """Initialize the message history with system message, context, task, and other initial messages"""
        self._add_message_with_tokens(self.system_prompt, message_type='init')

        if self.settings.message_context:
            context_message = HumanMessage(content='Context for the task' + self.settings.message_context)
            self._add_message_with_tokens(context_message, message_type='init')

        task_message = HumanMessage(
            content=f'Your ultimate task is: """{self.task}""". If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
        )
        self._add_message_with_tokens(task_message, message_type='init')

        if self.settings.sensitive_data:
            info = f'Here are placeholders for sensitive data: {list(self.settings.sensitive_data.keys())}'
            info += '\nTo use them, write <secret>the placeholder name</secret>'
            info_message = HumanMessage(content=info)
            self._add_message_with_tokens(info_message, message_type='init')

        # Add an example of interacting with a Flutter application
        placeholder_message = HumanMessage(content='Example output:')
        self._add_message_with_tokens(placeholder_message, message_type='init')

        example_tool_call = AIMessage(
            content='',
            tool_calls=[
                {
                    'name': 'AgentOutput',
                    'args': {
                        'current_state': {
                            'evaluation_previous_goal': """
                            Success - I successfully tapped on the 'Continue' button, which brought me to 
                            the main screen of the app. This is a successful step toward completing my task.
                            """.strip(),
                            'memory': """
                            I navigated through the app and found the login screen. I used the 'tap_element_by_index' 
                            tool to tap on element at index [15] labeled 'Continue' which took me to the main screen.
                            Currently at step 3/15.
                            """.strip(),
                            'next_goal': """
                            Looking at the widget structure of the current screen, I can see a SearchBar widget at 
                            index [8]. I'll use the 'enter_text' tool to search for the item I need to find.
                            """.strip(),
                        },
                        'action': [{'tap_element_by_index': {'index': 8}}],
                    },
                    'id': str(self.state.tool_id),
                    'type': 'tool_call',
                },
            ],
        )
        self._add_message_with_tokens(example_tool_call, message_type='init')
        self.add_tool_message(content='Flutter app initialized', message_type='init')

        placeholder_message = HumanMessage(content='[Your task history memory starts here]')
        self._add_message_with_tokens(placeholder_message)

        if self.settings.available_file_paths:
            filepaths_msg = HumanMessage(content=f'Here are file paths you can use: {self.settings.available_file_paths}')
            self._add_message_with_tokens(filepaths_msg, message_type='init')

    def add_new_task(self, new_task: str) -> None:
        """Update the task and add a message about it"""
        content = f'Your new ultimate task is: """{new_task}""". Take the previous context into account and finish your new ultimate task. '
        msg = HumanMessage(content=content)
        self._add_message_with_tokens(msg)
        self.task = new_task

    @time_execution_sync('--add_state_message')
    def add_state_message(
        self,
        node_state: NodeState,
        result: list[ActionResult] | None = None,
        step_info: AgentStepInfo | None = None,
        use_vision=True,
    ) -> None:
        """Add app state as human message"""

        # If results should be kept in memory, add them directly to history
        if result:
            for r in result:
                if r.include_in_memory:
                    if r.extracted_content:
                        msg = HumanMessage(content='Action result: ' + str(r.extracted_content))
                        self._add_message_with_tokens(msg)
                    if r.error:
                        # Format error message 
                        if r.error.endswith('\n'):
                            r.error = r.error[:-1]
                        # Get only last line of error for clarity
                        last_line = r.error.split('\n')[-1]
                        msg = HumanMessage(content='Action error: ' + last_line)
                        self._add_message_with_tokens(msg)
                    # Don't add result to next message if already in history
                    result = None  

        # Generate state message from NodeState
        state_message = AppMessagePrompt(
            node_state=node_state,
            result=result,
            include_attributes=self.settings.include_attributes,
            step_info=step_info,
        ).get_user_message(use_vision)
        
        self._add_message_with_tokens(state_message)

    def add_model_output(self, model_output: AgentOutput) -> None:
        """Add model output as AI message"""
        tool_calls = [
            {
                'name': 'AgentOutput',
                'args': model_output.model_dump(mode='json', exclude_unset=True),
                'id': str(self.state.tool_id),
                'type': 'tool_call',
            }
        ]

        msg = AIMessage(
            content='',
            tool_calls=tool_calls,
        )

        self._add_message_with_tokens(msg)
        # Add empty tool response to complete the exchange
        self.add_tool_message(content='')

    def add_plan(self, plan: str | None, position: int | None = None) -> None:
        """Add a planning analysis message"""
        if plan:
            msg = AIMessage(content=plan)
            self._add_message_with_tokens(msg, position)

    @time_execution_sync('--get_messages')
    def get_messages(self) -> list[BaseMessage]:
        """Get current message list"""
        messages = [m.message for m in self.state.history.messages]
        
        # Log token usage for debugging
        total_input_tokens = 0
        logger.debug(f'Messages in history: {len(self.state.history.messages)}:')
        for m in self.state.history.messages:
            total_input_tokens += m.metadata.tokens
            logger.debug(f'{m.message.__class__.__name__} - Token count: {m.metadata.tokens}')
        logger.debug(f'Total input tokens: {total_input_tokens}')

        return messages

    def _add_message_with_tokens(
        self, message: BaseMessage, position: int | None = None, message_type: str | None = None
    ) -> None:
        """Add message with token count metadata
        position: None for last, -1 for second last, etc.
        """
        # Filter out sensitive data
        if self.settings.sensitive_data:
            message = self._filter_sensitive_data(message)

        # Count tokens and add to history
        token_count = self._count_tokens(message)
        metadata = MessageMetadata(tokens=token_count, message_type=message_type)
        self.state.history.add_message(message, metadata, position)

    @time_execution_sync('--filter_sensitive_data')
    def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
        """Filter out sensitive data from the message"""

        def replace_sensitive(value: str) -> str:
            if not self.settings.sensitive_data:
                return value

            # Create a dictionary with all key-value pairs from sensitive_data where value is not None or empty
            valid_sensitive_data = {k: v for k, v in self.settings.sensitive_data.items() if v}

            # If there are no valid sensitive data entries, just return the original value
            if not valid_sensitive_data:
                logger.warning('No valid entries found in sensitive_data dictionary')
                return value

            # Replace all valid sensitive data values with their placeholder tags
            for key, val in valid_sensitive_data.items():
                value = value.replace(val, f'<secret>{key}</secret>')

            return value

        if isinstance(message.content, str):
            message.content = replace_sensitive(message.content)
        elif isinstance(message.content, list):
            for i, item in enumerate(message.content):
                if isinstance(item, dict) and 'text' in item:
                    item['text'] = replace_sensitive(item['text'])
                    message.content[i] = item
        return message

    def _count_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a message using a rough estimate"""
        tokens = 0
        if isinstance(message.content, list):
            for item in message.content:
                if 'image_url' in item:
                    tokens += self.settings.image_tokens
                elif isinstance(item, dict) and 'text' in item:
                    tokens += self._count_text_tokens(item['text'])
        else:
            msg = message.content
            if hasattr(message, 'tool_calls'):
                msg += str(message.tool_calls)  # type: ignore
            tokens += self._count_text_tokens(msg)
        return tokens

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        tokens = len(text) // self.settings.estimated_characters_per_token  # Rough estimate
        return tokens

    def cut_messages(self):
        """Get current message list, potentially trimmed to max tokens"""
        diff = self.state.history.current_tokens - self.settings.max_input_tokens
        if diff <= 0:
            return None

        msg = self.state.history.messages[-1]

        # If content is a list with image, remove the image first
        if isinstance(msg.message.content, list):
            text = ''
            for item in msg.message.content:
                if 'image_url' in item:
                    msg.message.content.remove(item)
                    diff -= self.settings.image_tokens
                    msg.metadata.tokens -= self.settings.image_tokens
                    self.state.history.current_tokens -= self.settings.image_tokens
                    logger.debug(
                        f'Removed image with {self.settings.image_tokens} tokens - total tokens now: '
                        f'{self.state.history.current_tokens}/{self.settings.max_input_tokens}'
                    )
                elif 'text' in item and isinstance(item, dict):
                    text += item['text']
            msg.message.content = text
            self.state.history.messages[-1] = msg

        if diff <= 0:
            return None

        # If still over token limit, remove text proportionally 
        proportion_to_remove = diff / msg.metadata.tokens
        if proportion_to_remove > 0.99:
            raise ValueError(
                f'Max token limit reached - history is too long - reduce the system prompt or task. '
                f'proportion_to_remove: {proportion_to_remove}'
            )
        logger.debug(
            f'Removing {proportion_to_remove * 100:.2f}% of the last message '
            f'({proportion_to_remove * msg.metadata.tokens:.2f} / {msg.metadata.tokens:.2f} tokens)'
        )

        content = msg.message.content
        characters_to_remove = int(len(content) * proportion_to_remove)
        content = content[:-characters_to_remove]

        # Remove old message and add new trimmed one
        self.state.history.remove_last_state_message()
        msg = HumanMessage(content=content)
        self._add_message_with_tokens(msg)

        last_msg = self.state.history.messages[-1]
        logger.debug(
            f'Added message with {last_msg.metadata.tokens} tokens - total tokens now: '
            f'{self.state.history.current_tokens}/{self.settings.max_input_tokens} - '
            f'total messages: {len(self.state.history.messages)}'
        )

    def _remove_last_state_message(self) -> None:
        """Remove last state message from history"""
        self.state.history.remove_last_state_message()

    def add_tool_message(self, content: str, message_type: str | None = None) -> None:
        """Add tool message to history"""
        msg = ToolMessage(content=content, tool_call_id=str(self.state.tool_id))
        self.state.tool_id += 1
        self._add_message_with_tokens(msg, message_type=message_type)
