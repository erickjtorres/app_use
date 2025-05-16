# app_use/controller/service.py
import asyncio
import logging
import inspect
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, create_model

from app_use.controller.views import ActionModel, ActionRegistry, RegisteredAction, ActionResult

# Setting up logging
logger = logging.getLogger(__name__)

Context = TypeVar('Context')

class Registry(Generic[Context]):
    """Service for registering and managing actions"""

    def __init__(self, exclude_actions: list[str] = None):
        self.registry = ActionRegistry()
        self.exclude_actions = exclude_actions if exclude_actions is not None else []

    def _create_param_model(self, function: Callable) -> type[BaseModel]:
        """Creates a Pydantic model from function signature"""
        sig = signature(function)
        params = {
            name: (param.annotation, ... if param.default == param.empty else param.default)
            for name, param in sig.parameters.items()
            if name != 'client' and name != 'all_nodes'  # Exclude common controller params
        }
        return create_model(
            f'{function.__name__}_parameters',
            __base__=ActionModel,
            **params,
        )

    def action(
        self,
        description: str,
        param_model: type[BaseModel] = None,
    ):
        """Decorator for registering actions"""

        def decorator(func: Callable):
            # Skip registration if action is in exclude_actions
            if func.__name__ in self.exclude_actions:
                return func

            # Create param model from function if not provided
            actual_param_model = param_model or self._create_param_model(func)

            # Wrap sync functions to make them async
            if not iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    return await asyncio.to_thread(func, *args, **kwargs)

                # Copy the signature and other metadata from the original function
                async_wrapper.__signature__ = signature(func)
                async_wrapper.__name__ = func.__name__
                async_wrapper.__annotations__ = func.__annotations__
                wrapped_func = async_wrapper
            else:
                wrapped_func = func

            action = RegisteredAction(
                name=func.__name__,
                description=description,
                function=wrapped_func,
                param_model=actual_param_model,
            )
            self.registry.actions[func.__name__] = action
            return func

        return decorator

    async def execute_action(
        self,
        action_name: str,
        params: dict,
        client=None,
        app=None,
        context: Context = None,
    ) -> Any:
        """Execute a registered action"""
        if action_name not in self.registry.actions:
            raise ValueError(f'Action {action_name} not found')

        action = self.registry.actions[action_name]
        try:
            # Create the validated Pydantic model
            validated_params = action.param_model(**params)

            # Check if the first parameter is a Pydantic model
            sig = signature(action.function)
            parameters = list(sig.parameters.values())
            is_pydantic = parameters and parameters[0].annotation != inspect.Parameter.empty and issubclass(parameters[0].annotation, BaseModel)
            parameter_names = [param.name for param in parameters]

            # Prepare arguments based on parameter type
            extra_args = {}
            if 'context' in parameter_names:
                extra_args['context'] = context
            if 'client' in parameter_names:
                extra_args['client'] = client
            if 'app' in parameter_names:
                extra_args['app'] = app

            if is_pydantic:
                return await action.function(validated_params, **extra_args)
            return await action.function(**validated_params.model_dump(), **extra_args)

        except Exception as e:
            logger.error(f'Error executing action {action_name}: {str(e)}')
            raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

    def create_action_model(self, include_actions: list[str] = None) -> type[ActionModel]:
        """Creates a Pydantic model from registered actions"""
        available_actions = {}
        for name, action in self.registry.actions.items():
            if include_actions is not None and name not in include_actions:
                continue
            available_actions[name] = action

        fields = {
            name: (
                Optional[action.param_model],
                Field(default=None, description=action.description),
            )
            for name, action in available_actions.items()
        }

        return create_model('ActionModel', __base__=ActionModel, **fields)

    def get_prompt_description(self) -> str:
        """Get a description of all actions for the prompt"""
        return self.registry.get_prompt_description()