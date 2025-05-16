# app_use/controller/views.py
from collections.abc import Callable
from typing import Any, Optional, List

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Action Input Models
class ActionModel(BaseModel):
    """Base model for dynamically created action models"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_index(self) -> int | None:
        """Get the index of the action if it exists"""
        params = self.model_dump(exclude_unset=True).values()
        if not params:
            return None
        for param in params:
            if param is not None and 'index' in param:
                return param['index']
        return None

    def set_index(self, index: int):
        """Overwrite the index of the action"""
        # Get the action name and params
        action_data = self.model_dump(exclude_unset=True)
        action_name = next(iter(action_data.keys()))
        action_params = getattr(self, action_name)

        # Update the index directly on the model
        if hasattr(action_params, 'index'):
            action_params.index = index


class RegisteredAction(BaseModel):
    """Model for a registered action"""

    name: str
    description: str
    function: Callable
    param_model: type[BaseModel]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def prompt_description(self) -> str:
        """Get a description of the action for the prompt"""
        skip_keys = ['title']
        s = f'{self.description}: \n'
        s += '{' + str(self.name) + ': '
        s += str(
            {
                k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
                for k, v in self.param_model.model_json_schema()['properties'].items()
            }
        )
        s += '}'
        return s


class ActionRegistry(BaseModel):
    """Model representing the action registry"""

    actions: dict[str, RegisteredAction] = {}

    def get_prompt_description(self) -> str:
        """Get a description of all actions for the prompt"""
        return '\n'.join(action.prompt_description() for action in self.actions.values())


class ActionResult(BaseModel):
    """Result of an action execution"""
    
    is_done: bool = False
    success: bool = True
    error: str | None = None
    extracted_content: str | None = None
    include_in_memory: bool = False


# Flutter App Specific Action Models

class ClickWidgetAction(BaseModel):
    """Action model for clicking a widget by its unique ID"""
    unique_id: int

class EnterTextAction(BaseModel):
    """Action model for entering text into a widget by its unique ID"""
    unique_id: int
    text: str

class ScrollIntoViewAction(BaseModel):
    """Action model for scrolling a widget into view by its unique ID"""
    unique_id: int

class ScrollUpOrDownAction(BaseModel):
    """Action model for scrolling a widget up or down"""
    unique_id: int
    direction: str = "down"

class ScrollExtendedAction(BaseModel):
    """Action model for performing an extended scroll with more parameters"""
    unique_id: int
    direction: str = "down"
    dx: int = 0
    dy: int = 100
    duration_microseconds: int = 300000
    frequency: int = 60

class FindScrollableAncestorAction(BaseModel):
    """Action model for finding the closest scrollable ancestor of a widget"""
    unique_id: int

class FindScrollableDescendantAction(BaseModel):
    """Action model for finding the first scrollable descendant of a widget"""
    unique_id: int

class GetAppStateAction(BaseModel):
    """Action model for getting the current application state"""
    model_config = ConfigDict(extra='allow')
    
    @model_validator(mode='before')
    def ignore_all_inputs(cls, values):
        # No matter what the user sends, discard it and return empty.
        return {}

class DoneAction(BaseModel):
    """Action model for completing a task with a result"""
    text: str
    success: bool = True

# Helper model for models that require no parameters
class NoParamsAction(BaseModel):
    """
    Accepts absolutely anything in the incoming data
    and discards it, so the final parsed model is empty.
    """
    model_config = ConfigDict(extra='allow')
    
    @model_validator(mode='before')
    def ignore_all_inputs(cls, values):
        # No matter what the user sends, discard it and return empty.
        return {}