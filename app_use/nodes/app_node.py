# widget_node.py
import hashlib
import logging

logger = logging.getLogger("AppNode")

class AppNode:
    def __init__(self, unique_id, widget_type, is_interactive, properties, parent_node=None, text=None, key=None):
        self.unique_id = unique_id
        self.widget_type = widget_type
        self.is_interactive = is_interactive
        self.properties = properties
        self.parent_node = parent_node
        self.child_nodes = []
        self.previous_sibling = None
        self.next_sibling = None
        self.text = text
        self.key = key

    def get_node_path(self):
        """Return the full path to this node in the tree"""
        path = []
        current = self
        while current:
            path.insert(0, f"{current.widget_type}({current.unique_id})")
            current = current.parent_node
        return ' > '.join(path)

    def __repr__(self):
        return (f"AppNode(type={self.widget_type}, "
                f"interactive={self.is_interactive}, unique_id={self.unique_id}, "
                f"text={self.text}, key={self.key}, "
                f"parent={self.parent_node.unique_id if self.parent_node else None}, "
                f"children={[c.unique_id for c in self.child_nodes]})")


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