# app_use/controller/controller.py
import asyncio
import logging
from typing import Generic, TypeVar, Any, List

from pydantic import BaseModel

from app_use.controller.registry.service import Registry
from app_use.controller.views import (
    ActionModel, 
    ActionResult, 
    DoneAction,
    ClickWidgetAction,
    EnterTextAction,
    ScrollIntoViewAction,
    ScrollUpOrDownAction,
    ScrollExtendedAction,
    FindScrollableAncestorAction,
    FindScrollableDescendantAction,
    GetAppStateAction
)
from app_use.nodes.app_node import AppNode
from app_use.app.app import App

logger = logging.getLogger(__name__)

Context = TypeVar('Context')

class Controller(Generic[Context]):
    def __init__(
        self,
        exclude_actions: list[str] = None,
        output_model: type[BaseModel] = None,
    ):
        self.registry = Registry[Context](exclude_actions)

        """Register all default app actions"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):
                success: bool = True
                data: output_model

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False)',
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()
                return ActionResult(is_done=True, success=params.success, extracted_content=str(output_dict))
        else:
            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False)',
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        # Register click widget action
        @self.registry.action(
            'Click a widget element by its unique ID',
            param_model=ClickWidgetAction,
        )
        async def click_widget(params: ClickWidgetAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            success = app.click_widget_by_unique_id(all_nodes, params.unique_id)
            
            if success:
                msg = f"🖱️ Clicked widget with unique ID {params.unique_id}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                error_msg = f"Failed to click widget with unique ID {params.unique_id}"
                return ActionResult(error=error_msg, include_in_memory=True)

        # Register enter text action
        @self.registry.action(
            'Enter text into a widget element by its unique ID',
            param_model=EnterTextAction,
        )
        async def enter_text(params: EnterTextAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            success = app.enter_text_with_unique_id(all_nodes, params.unique_id, params.text)
            
            if success:
                msg = f"⌨️ Entered text '{params.text}' into widget with unique ID {params.unique_id}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                error_msg = f"Failed to enter text into widget with unique ID {params.unique_id}"
                return ActionResult(error=error_msg, include_in_memory=True)

        # Register scroll into view action
        @self.registry.action(
            'Scroll a widget into view by its unique ID',
            param_model=ScrollIntoViewAction,
        )
        async def scroll_into_view(params: ScrollIntoViewAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            success = app.scroll_into_view(all_nodes, params.unique_id)
            
            if success:
                msg = f"🔍 Scrolled widget with unique ID {params.unique_id} into view"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                error_msg = f"Failed to scroll widget with unique ID {params.unique_id} into view"
                return ActionResult(error=error_msg, include_in_memory=True)

        # Register scroll up/down action
        @self.registry.action(
            'Scroll a widget up or down',
            param_model=ScrollUpOrDownAction,
        )
        async def scroll_up_or_down(params: ScrollUpOrDownAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            # Validate direction
            if params.direction not in ["up", "down"]:
                error_msg = f"Invalid scroll direction: {params.direction}. Must be 'up' or 'down'."
                return ActionResult(error=error_msg, include_in_memory=True)
                
            success = app.scroll_up_or_down(all_nodes, params.unique_id, params.direction)
            
            if success:
                msg = f"🔍 Scrolled {params.direction} with widget unique ID {params.unique_id}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                error_msg = f"Failed to scroll {params.direction} with widget unique ID {params.unique_id}"
                return ActionResult(error=error_msg, include_in_memory=True)

        # Register extended scroll action
        @self.registry.action(
            'Perform an extended scroll with more parameters on a widget',
            param_model=ScrollExtendedAction,
        )
        async def scroll_extended(params: ScrollExtendedAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            # Validate direction
            if params.direction not in ["up", "down"]:
                error_msg = f"Invalid scroll direction: {params.direction}. Must be 'up' or 'down'."
                return ActionResult(error=error_msg, include_in_memory=True)
                
            success = app.scroll_up_or_down_extended(
                all_nodes, 
                params.unique_id, 
                params.direction, 
                params.dx, 
                params.dy, 
                params.duration_microseconds,
                params.frequency
            )
            
            if success:
                msg = f"🔍 Performed extended {params.direction} scroll on widget {params.unique_id} with parameters: dx={params.dx}, dy={params.dy}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                error_msg = f"Failed to perform extended {params.direction} scroll on widget {params.unique_id}"
                return ActionResult(error=error_msg, include_in_memory=True)

        # Register action to find scrollable ancestor
        @self.registry.action(
            'Find the closest scrollable ancestor of a widget',
            param_model=FindScrollableAncestorAction,
        )
        async def find_scrollable_ancestor(params: FindScrollableAncestorAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            # Find the target node first
            target_node = None
            for node in all_nodes:
                if node.unique_id == params.unique_id:
                    target_node = node
                    break
                    
            if not target_node:
                error_msg = f"No widget found with unique_id: {params.unique_id}"
                return ActionResult(error=error_msg, include_in_memory=True)
            
            scrollable_ancestor = app.find_ancestor_with_scroll(target_node)
            
            if scrollable_ancestor:
                msg = f"Found scrollable ancestor with unique ID {scrollable_ancestor.unique_id} and type {scrollable_ancestor.widget_type}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                msg = f"No scrollable ancestor found for widget with unique ID {params.unique_id}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
                
        # Register action to find scrollable descendant
        @self.registry.action(
            'Find the first scrollable descendant of a widget',
            param_model=FindScrollableDescendantAction,
        )
        async def find_scrollable_descendant(params: FindScrollableDescendantAction, app: App):
            # Get the current app state
            all_nodes = app.get_app_state()
            
            # Find the target node first
            target_node = None
            for node in all_nodes:
                if node.unique_id == params.unique_id:
                    target_node = node
                    break
                    
            if not target_node:
                error_msg = f"No widget found with unique_id: {params.unique_id}"
                return ActionResult(error=error_msg, include_in_memory=True)
            
            scrollable_descendant = app.find_descendant_with_scroll(target_node)
            
            if scrollable_descendant:
                msg = f"Found scrollable descendant with unique ID {scrollable_descendant.unique_id} and type {scrollable_descendant.widget_type}"
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                msg = f"No scrollable descendant found for widget with unique ID {params.unique_id}"
                return ActionResult(extracted_content=msg, include_in_memory=True)

        # Register action to get app state
        @self.registry.action(
            'Get the current application state with all widget nodes',
            param_model=GetAppStateAction,
        )
        async def get_app_state(params: GetAppStateAction, app: App):
            all_nodes = app.get_app_state()
            
            # Format nodes for display
            node_info = []
            for node in all_nodes:
                info = {
                    "unique_id": node.unique_id,
                    "widget_type": node.widget_type,
                    "is_interactive": node.is_interactive,
                    "text": node.text,
                    "key": node.key,
                    "parent": node.parent_node.unique_id if node.parent_node else None,
                }
                node_info.append(info)
            
            msg = f"Retrieved app state with {len(all_nodes)} nodes:\n{str(node_info)}"
            return ActionResult(extracted_content=msg, include_in_memory=True)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions"""
        return self.registry.action(description, **kwargs)

    # Act --------------------------------------------------------------------

    async def act(
        self,
        action: ActionModel,
        app: App,
        context: Context = None,
    ) -> ActionResult:
        """Execute an action"""
        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        app=app,
                        context=context,
                    )

            if isinstance(result, str):
                return ActionResult(extracted_content=result)
            elif isinstance(result, ActionResult):
                return result
            elif result is None:
                return ActionResult()
            else:
                raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            raise e