# widget_node.py
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("AppNode")


class AppBaseNode:
    """Base class shared by all nodes in the widget tree."""

    NODE_TYPE = "BASE_NODE"  # Default type

    def __init__(self, unique_id: int, parent: Optional["AppElementNode"] = None, key: Optional[str] = None):
        self.unique_id: int = unique_id
        # Store parent reference using the same name as before for backwards-compat.
        self.parent: Optional["AppElementNode"] = parent
        # Alias for legacy code that still expects *parent_node*
        self.parent_node: Optional["AppElementNode"] = parent
        # Default widget_type that can be overridden by subclasses
        self.widget_type: str = self.NODE_TYPE
        # Flutter widget key
        self.key: Optional[str] = key
        # Node hierarchy relationships
        self.child_nodes: List["AppBaseNode"] = []
        self.previous_sibling: Optional["AppBaseNode"] = None
        self.next_sibling: Optional["AppBaseNode"] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def is_visible(self) -> bool:
        """Visibility placeholder – assumed always visible for now."""
        return True

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_json(self) -> dict:  # noqa: D401
        """Return a JSON-serialisable representation of *self*."""
        return {
            "id": self.unique_id,
            "type": self.NODE_TYPE,
            "widget_type": self.widget_type,
            "key": self.key,
            "parent": self.parent.unique_id if self.parent else None,
            "children": [c.to_json() for c in self.child_nodes],
            "previous_sibling": self.previous_sibling.unique_id if self.previous_sibling else None,
            "next_sibling": self.next_sibling.unique_id if self.next_sibling else None
        }


# ======================================================================
#                            Text  Node
# ======================================================================

class AppTextNode(AppBaseNode):
    """A leaf node that only contains text."""

    NODE_TYPE = "TEXT_NODE"

    def __init__(self, unique_id: int, text: str, parent: Optional["AppElementNode"] = None, key: Optional[str] = None):
        super().__init__(unique_id, parent, key)
        self.text: str = text
        self.widget_type: str = "Text"  # Use 'Text' for compatibility with filtering

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def parent_is_interactive(self) -> bool:
        """True when any ancestor element is *interactive* or has interactive children."""
        # First check if any child nodes are interactive (for nested text nodes)
        for child in self.child_nodes:
            if getattr(child, "is_interactive", False):
                return True
                
        # Then check ancestors
        current = self.parent
        while current is not None:
            if getattr(current, "is_interactive", False):
                return True
            current = current.parent
        return False
        
    @property
    def is_interactive(self) -> bool:
        """Text nodes are considered interactive if their parent is interactive."""
        return self.parent_is_interactive()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_json(self) -> dict:  # noqa: D401
        base_json = super().to_json()
        text_json = {
            "text": self.text,
            "interactive": self.is_interactive,
        }
        # Merge the dictionaries
        return {**base_json, **text_json}

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"AppTextNode(id={self.unique_id}, text={self.text!r})"


# ======================================================================
#                           Element  Node
# ======================================================================

class AppElementNode(AppBaseNode):
    """Represents any widget element other than a raw text leaf."""

    NODE_TYPE = "ELEMENT_NODE"

    def __init__(
        self,
        unique_id: int,
        widget_type: str,
        is_interactive: bool,
        properties: dict,
        parent_node: Optional["AppElementNode"] = None,
        text: Optional[str] = None,
        key: Optional[str] = None,
    ):
        super().__init__(unique_id, parent_node, key)
        # Override the base widget_type with the specific one for this element
        self.widget_type: str = widget_type
        self.is_interactive: bool = bool(is_interactive)
        self.properties: dict = properties or {}
        self.text: Optional[str] = text
        self.key: Optional[str] = key

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------
    def add_child(self, child: AppBaseNode) -> None:
        self.child_nodes.append(child)
        # Maintain both *parent* and legacy *parent_node* links
        child.parent = self
        child.parent_node = self  # All nodes now have parent_node

    def get_node_path(self) -> str:
        """Return a human-readable path from the root to *self*."""
        path: List[str] = []
        current: Optional[AppBaseNode] = self
        while current is not None:
            if isinstance(current, AppElementNode):
                name = f"{current.widget_type}({current.unique_id})"
            else:  # Text node
                name = f"Text({current.unique_id})"
            path.insert(0, name)
            current = current.parent
        return " > ".join(path)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_json(self) -> dict:  # noqa: D401
        base_json = super().to_json()
        element_json = {
            "interactive": self.is_interactive,
            "text": self.text,
            "properties": self.properties,
        }
        # Merge the dictionaries
        return {**base_json, **element_json}

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AppElementNode(type={self.widget_type}, interactive={self.is_interactive}, "
            f"id={self.unique_id}, text={self.text}, key={self.key}, "
            f"children={[c.unique_id for c in self.child_nodes]})"
        )


# ----------------------------------------------------------------------
# Backwards-compat:   Keep *AppNode* name alive until callers migrate.
# ----------------------------------------------------------------------
AppNode = AppElementNode  # type: ignore


# ----------------------------------------------------------------------
# Selector map & NodeState container
# ----------------------------------------------------------------------

SelectorMap = Dict[int, AppBaseNode]


@dataclass
class NodeState:
    """State wrapper containing the root *element_tree* and a *selector_map*."""

    element_tree: AppElementNode
    selector_map: SelectorMap = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_json(self) -> dict:  # noqa: D401
        return {
            "element_tree": self.element_tree.to_json(),
            "selector_map": {uid: node.to_json() for uid, node in self.selector_map.items()},
        }


class AppNodeUtils:
    @staticmethod
    def find_node_by_unique_id(nodes, unique_id):
        """Find a node by its unique ID"""
        for node in nodes:
            if node.unique_id == unique_id:
                return node
        return None

    @staticmethod
    def find_node_by_key(nodes, key_name):
        """Find a node by its Flutter key value"""
        for node in nodes:
            if node.key == key_name:
                return node
        return None

    @staticmethod
    def find_interactive_nodes(nodes):
        """Find interactive nodes in the tree"""
        return [node for node in nodes if node.is_interactive]

    @staticmethod
    def find_nodes_by_key(nodes, key_name):
        """Find nodes by their Flutter key value"""
        return [
            node for node in nodes 
            if node.key and key_name.lower() in node.key.lower()
        ]

    @staticmethod
    def find_nodes_by_type(nodes, type_str):
        """Find nodes by widget type"""
        return [node for node in nodes if type_str.lower() in node.widget_type.lower()]

    @staticmethod
    def find_nodes_by_text(nodes, text_str):
        """Find nodes by their visible text content"""
        return [
            node for node in nodes 
            if node.text and text_str.lower() in node.text.lower()
        ]

    @staticmethod
    def find_nodes_by_description(nodes, description):
        """Find nodes by description in their properties"""
        return [
            node for node in nodes 
            if 'description' in node.properties and 
               description.lower() in str(node.properties['description']).lower()
        ]

    @staticmethod
    def categorize_ui_elements(nodes):
        """Categorize UI elements by their function"""
        categorized = {
            'navigation': [],
            'input': [],
            'buttons': [],
            'text': [],
            'images': [],
            'cards': [],
            'lists': [],
            'tiles': [],
            'containers': [],
            'other': [],
        }

        for node in nodes:
            widget_type = node.widget_type.lower()
            description = str(node.properties.get('description', '')).lower()
            
            # Categorize based on type and description
            if any(nav in widget_type or nav in description for nav in 
                   ['appbar', 'navigation', 'bottombar', 'drawer', 'home', 'profile', 'menu', 'nav']):
                categorized['navigation'].append(node)
            elif any(inp in widget_type or inp in description for inp in 
                     ['textfield', 'input', 'search', 'form']):
                categorized['input'].append(node)
            elif any(btn in widget_type or btn in description for btn in 
                     ['button', 'gesture', 'inkwell']):
                categorized['buttons'].append(node)
            elif 'text' in widget_type and 'field' not in widget_type:
                categorized['text'].append(node)
            elif any(img in widget_type or img in description for img in 
                     ['image', 'icon', 'picture']):
                categorized['images'].append(node)
            elif any(crd in widget_type or crd in description for crd in 
                     ['card', 'product']):
                categorized['cards'].append(node)
            elif any(lst in widget_type or lst in description for lst in 
                     ['list', 'grid', 'view']):
                categorized['lists'].append(node)
            elif any(tile in widget_type or tile in description for tile in 
                     ['tile', 'item']):
                categorized['tiles'].append(node)
            elif any(container in widget_type for container in 
                     ['container', 'box', 'padding', 'column', 'row', 'stack']):
                categorized['containers'].append(node)
            else:
                categorized['other'].append(node)
                
        return categorized

    @staticmethod
    def extract_text_content(nodes):
        """Find all text content in the UI"""
        text_content = {}
        
        for node in nodes:
            # Use text property if available
            if node.text:
                text_content[f'node_{node.unique_id}'] = node.text
            
            # Text widgets with description
            if 'Text' in node.widget_type and 'description' in node.properties:
                text = str(node.properties['description'])
                text_content[f'text_{node.unique_id}'] = text
            
            # Text preview available
            if 'textPreview' in node.properties:
                text_content[f'preview_{node.unique_id}'] = node.properties['textPreview']
                
            # Data property
            if 'data' in node.properties and isinstance(node.properties['data'], str):
                text_content[f'data_{node.unique_id}'] = node.properties['data']
                
            # Label property
            if 'label' in node.properties and isinstance(node.properties['label'], str):
                text_content[f'label_{node.unique_id}'] = node.properties['label']
                
            # Hint property  
            if 'hint' in node.properties and isinstance(node.properties['hint'], str):
                text_content[f'hint_{node.unique_id}'] = node.properties['hint']
                
            # Extract text from descriptions that might contain actual UI text
            if ('description' in node.properties and 
                'Text' not in node.widget_type and 
                isinstance(node.properties['description'], str)):
                
                desc = node.properties['description']
                # Only include if it looks like actual content
                if not any(c in desc for c in ['_', '[', '{', '<']) and 1 < len(desc) < 50:
                    text_content[f'desc_{node.unique_id}'] = desc
        
        return text_content