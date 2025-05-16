from app_use.nodes.node_tree_builder import NodeTreeBuilder
from dart_vm_client import DartVmServiceManager, DartVmServiceClient
import atexit


class App:
    def __init__(self, client=None, vm_service_uri=None):
        """
        Initialize the App with a client or create a new client and connect to a VM service URI.
        
        Args:
            client: An existing DartVmServiceClient instance, or None to create a new one
            vm_service_uri: The WebSocket URI of the VM service to connect to (if client is None)
        
        Raises:
            ValueError: If both client and vm_service_uri are None, or if connection fails
        """
        # Handle client setup
        self._manage_client = False
        
        if client is None:
            if vm_service_uri is None:
                raise ValueError("Either client or vm_service_uri must be provided")

            # Start the Dart VM service manager
            self.service_manager = DartVmServiceManager(port=50052)
            if not self.service_manager.start():
                raise ValueError("Failed to start Dart VM Service Manager on port 50052")

            # Create a client that connects to the running service
            print(f"Using DartVmServiceClient to connect to {vm_service_uri}")
            self.client = DartVmServiceClient("localhost:50052")
            response = self.client.connect(vm_service_uri)
            if not hasattr(response, 'success') or not response.success:
                error_msg = getattr(response, 'message', 'Unknown error')
                # Clean up on failure
                self.client.close()
                self.service_manager.stop()
                raise ValueError(f"Failed to connect to Flutter app: {error_msg}")

            self._manage_client = True
            atexit.register(self.close)
        else:
            self.client = client
            
        self.app_state = {}
        
    def get_app_state(self):
        """
        Get the current state of the app
        """
        
        builder = NodeTreeBuilder(self.client)

        all_nodes = builder.build_widget_tree("flutter")
        return all_nodes
    
    def enter_text_with_unique_id(self, all_nodes, unique_id, text):
        """
        Finds a widget by its unique_id and triggers a text entry action, 
        prioritizing enter_text_by_ancestor_and_descendant method
        
        Args:
            all_nodes: List of AppNode objects from the widget tree
            unique_id: The unique identifier of the widget to enter text
            text: The text to enter
        
        Returns:
            Boolean indicating success or failure
        """
        # Find the node with the matching unique_id
        target_node = None
        for node in all_nodes:
            if node.unique_id == unique_id:
                target_node = node
                break
            
        if not target_node:
            print(f"No widget found with unique_id: {unique_id}")
            return False
        
        print(f"Attempting to enter text in {target_node.widget_type}")
        
        # Try by Flutter key if available
        if target_node.key:
            try:
                print(f"Trying enter text by Flutter key: {target_node.key}")
                response = self.client.enter_text_by_key(target_node.key, text)
                    
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully entered text using Flutter key")
                    return True
            except Exception as e:
                print(f"Error entering text by Flutter key: {e}")
        
        # Try by text content if available
        if target_node.text:
            try:
                print(f"Trying enter text by text: '{target_node.text}'")
                response = self.client.enter_text_by_text(target_node.text, text)
                        
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully entered text using text content")
                    return True
            except Exception as e:
                print(f"Error entering text by text: {e}")
        
        # Try with ancestor-descendant approach
        if target_node.parent_node:
            # Get ancestor type (parent widget type)
            ancestor_type = target_node.parent_node.widget_type.split(' ')[0]  # Get main type without description
            # Get descendant type (current widget type)
            descendant_type = target_node.widget_type.split(' ')[0]  # Get main type without description
            
            try:
                print(f"Trying enter text by ancestor-descendant: {ancestor_type} -> {descendant_type}")
                response = self.client.enter_text_by_ancestor_and_descendant(
                    ancestor_type, descendant_type, text
                )
                        
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully entered text using ancestor-descendant method")
                    return True
            except Exception as e:
                print(f"Error with ancestor-descendant text entry: {e}")
        
        # Try by widget type
        try:
            widget_type = target_node.widget_type.split(' ')[0]  # Use main widget type
            print(f"Trying enter text by widget type: {widget_type}")
            
            response = self.client.enter_text_by_type(widget_type, text)
                    
            if hasattr(response, 'success') and response.success:
                print(f"Successfully entered text using widget type")
                return True
        except Exception as e:
            print(f"Error entering text by widget type: {e}")
            
        # Try tooltip if available
        if hasattr(target_node, 'tooltip') and target_node.tooltip:
            try:
                print(f"Trying enter text by tooltip: {target_node.tooltip}")
                response = self.client.enter_text_by_tooltip(target_node.tooltip, text)
                
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully entered text using tooltip")
                    return True
            except Exception as e:
                print(f"Error entering text by tooltip: {e}")
        
        # Try generic enter_text method as a fallback
        try:
            print(f"Trying generic enter_text method as fallback")
            identifier = target_node.key or target_node.text or target_node.widget_type
            response = self.client.enter_text(identifier, text)
            
            if hasattr(response, 'success') and response.success:
                print(f"Successfully entered text using generic method")
                return True
        except Exception as e:
            print(f"Error with generic text entry method: {e}")
            
        print(f"Failed to enter text in widget with unique_id: {unique_id}")
        return False
    
    
    def click_widget_by_unique_id(self, all_nodes, unique_id: int):
        """
        Finds a widget by its unique_id and triggers a tap action, 
        prioritizing tap_widget_by_ancestor_and_descendant method
        
        Args:
            client: DartVmServiceClient instance
            all_nodes: List of AppNode objects from the widget tree
            unique_id: The unique identifier of the widget to click
        
        Returns:
            Boolean indicating success or failure
        """
        # Find the node with the matching unique_id
        target_node = None
        for node in all_nodes:
            if node.unique_id == unique_id:
                target_node = node
                break
            
        if not target_node:
            print(f"No widget found with unique_id: {unique_id}")
            return False
        
        # Check if the node is interactive
        if not target_node.is_interactive:
            print(f"Widget with unique_id {unique_id} may not be interactive")
            # Continue anyway as our check might not be perfect
        
        print(f"Attempting to click on {target_node.widget_type}")
        
        
        # Try by Flutter key if available
        if target_node.key:
            try:
                print(f"Trying tap by Flutter key: {target_node.key}")
                response = self.client.tap_widget_by_key(target_node.key)
                if hasattr(response, 'success') and response.success:
                    print("Successfully tapped using Flutter key")
                    return True
            except Exception as e:
                print(f"Error tapping by Flutter key: {e}")
        
        # Try by text content if available
        if target_node.text:
            try:
                print(f"Trying tap by text: '{target_node.text}'")
                response = self.client.tap_widget_by_text(target_node.text)
                if hasattr(response, 'success') and response.success:
                    print("Successfully tapped using text content")
                    return True
            except Exception as e:
                print(f"Error tapping by text: {e}")
        
        
        # Try with ancestor-descendant approach (preferred method)
        if target_node.parent_node:
            # Get ancestor type (parent widget type)
            ancestor_type = target_node.parent_node.widget_type.split(' ')[0]  # Get main type without description
            # Get descendant type (current widget type)
            descendant_type = target_node.widget_type.split(' ')[0]  # Get main type without description
            
            try:
                print(f"Trying tap by ancestor-descendant: {ancestor_type} -> {descendant_type}")
                response = self.client.tap_widget_by_ancestor_and_descendant(ancestor_type, descendant_type)
                if hasattr(response, 'success') and response.success:
                    print("Successfully tapped using ancestor-descendant method")
                    return True
            except Exception as e:
                print(f"Error with ancestor-descendant tap: {e}")
                
        
        # Try by widget type
        try:
            print(f"Trying tap by widget type: {target_node.widget_type}")
            response = self.client.tap_widget_by_type(target_node.widget_type.split(' ')[0])  # Use main widget type
            if hasattr(response, 'success') and response.success:
                print("Successfully tapped using widget type")
                return True
        except Exception as e:
            print(f"Error tapping by widget type: {e}")
        
        
        # If all attempts fail, check if there's an interactive parent we could tap instead
        if target_node.parent_node and target_node.parent_node.is_interactive:
            print(f"Current widget couldn't be tapped, trying parent: {target_node.parent_node.widget_type}")
            return self.click_widget_by_unique_id(all_nodes, target_node.parent_node.unique_id)
        
        print(f"Failed to click on widget with unique_id: {unique_id}")
        return False

    def scroll_into_view(self, all_nodes, unique_id):
        """
        Scroll a widget into view by trying different methods in order of expected reliability.

        Args:
            all_nodes: List of AppNode objects from the widget tree
            unique_id: The unique identifier of the widget to scroll into view

        Returns:
            Boolean indicating success or failure
        """
        # Find the node with the matching unique_id
        target_node = None
        for node in all_nodes:
            if node.unique_id == unique_id:
                target_node = node
                break

        if not target_node:
            print(f"No widget found with unique_id: {unique_id}")
            return False

        print(f"Attempting to scroll into view: {target_node.widget_type}")

        # Try by Flutter key if available
        if target_node.key:
            try:
                print(f"Trying scroll into view by key: {target_node.key}")
                response = self.client.scroll_into_view_by_key(target_node.key)
                if hasattr(response, 'success') and response.success:
                    print("Successfully scrolled into view using key")
                    return True
            except Exception as e:
                print(f"Error scrolling into view by key: {e}")

        # Try by text content if available
        if target_node.text:
            try:
                print(f"Trying scroll into view by text: '{target_node.text}'")
                response = self.client.scroll_into_view_by_text(target_node.text)
                if hasattr(response, 'success') and response.success:
                    print("Successfully scrolled into view using text")
                    return True
            except Exception as e:
                print(f"Error scrolling into view by text: {e}")

        # Try by widget type
        try:
            print(f"Trying scroll into view by type: {target_node.widget_type}")
            response = self.client.scroll_into_view_by_type(target_node.widget_type)
            if hasattr(response, 'success') and response.success:
                print("Successfully scrolled into view using type")
                return True
        except Exception as e:
            print(f"Error scrolling into view by type: {e}")

        # Try by tooltip if available
        if 'tooltip' in target_node.properties:
            try:
                tooltip = target_node.properties['tooltip']
                print(f"Trying scroll into view by tooltip: {tooltip}")
                response = self.client.scroll_into_view_by_tooltip(tooltip)
                if hasattr(response, 'success') and response.success:
                    print("Successfully scrolled into view using tooltip")
                    return True
            except Exception as e:
                print(f"Error scrolling into view by tooltip: {e}")

        # Try with ancestor-descendant approach
        if target_node.parent_node:
            # Get ancestor type (parent widget type)
            ancestor_type = target_node.parent_node.widget_type.split(' ')[0]  # Get main type without description
            # Get descendant type (current widget type)
            descendant_type = target_node.widget_type.split(' ')[0]  # Get main type without description

            try:
                print(f"Trying scroll into view by ancestor-descendant: {ancestor_type} -> {descendant_type}")
                response = self.client.scroll_into_view_by_ancestor_and_descendant(ancestor_type, descendant_type)
                if hasattr(response, 'success') and response.success:
                    print("Successfully scrolled into view using ancestor-descendant")
                    return True
            except Exception as e:
                print(f"Error with ancestor-descendant scroll into view: {e}")

        print(f"Failed to scroll into view for widget with unique_id: {unique_id}")
        return False

    def scroll_up_or_down(self, all_nodes, unique_id, direction="down"):
        """
        Scroll a widget up or down by trying different methods in order of expected reliability.

        Args:
            all_nodes: List of AppNode objects from the widget tree
            unique_id: The unique identifier of the widget to scroll
            direction: "up" or "down" scroll direction

        Returns:
            Boolean indicating success or failure
        """
        # Find the node with the matching unique_id
        target_node = None
        for node in all_nodes:
            if node.unique_id == unique_id:
                target_node = node
                break

        if not target_node:
            print(f"No widget found with unique_id: {unique_id}")
            return False

        print(f"Attempting to scroll {direction}: {target_node.widget_type}")

        # Try by Flutter key if available
        if target_node.key:
            try:
                print(f"Trying scroll {direction} by key: {target_node.key}")
                if direction == "down":
                    response = self.client.scroll_down_by_key(target_node.key)
                else:
                    response = self.client.scroll_up_by_key(target_node.key)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using key")
                    return True
            except Exception as e:
                print(f"Error scrolling {direction} by key: {e}")

        # Try by text content if available
        if target_node.text:
            try:
                print(f"Trying scroll {direction} by text: '{target_node.text}'")
                if direction == "down":
                    response = self.client.scroll_down_by_text(target_node.text)
                else:
                    response = self.client.scroll_up_by_text(target_node.text)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using text")
                    return True
            except Exception as e:
                print(f"Error scrolling {direction} by text: {e}")

        # Try by widget type
        try:
            print(f"Trying scroll {direction} by type: {target_node.widget_type}")
            if direction == "down":
                response = self.client.scroll_down_by_type(target_node.widget_type)
            else:
                response = self.client.scroll_up_by_type(target_node.widget_type)
            if hasattr(response, 'success') and response.success:
                print(f"Successfully scrolled {direction} using type")
                return True
        except Exception as e:
            print(f"Error scrolling {direction} by type: {e}")

        # Try by tooltip if available
        if 'tooltip' in target_node.properties:
            try:
                tooltip = target_node.properties['tooltip']
                print(f"Trying scroll {direction} by tooltip: {tooltip}")
                if direction == "down":
                    response = self.client.scroll_down_by_tooltip(tooltip)
                else:
                    response = self.client.scroll_up_by_tooltip(tooltip)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using tooltip")
                    return True
            except Exception as e:
                print(f"Error scrolling {direction} by tooltip: {e}")

        # Try with ancestor-descendant approach
        if target_node.parent_node:
            # Get ancestor type (parent widget type)
            ancestor_type = target_node.parent_node.widget_type.split(' ')[0]  # Get main type without description
            # Get descendant type (current widget type)
            descendant_type = target_node.widget_type.split(' ')[0]  # Get main type without description

            try:
                print(f"Trying scroll {direction} by ancestor-descendant: {ancestor_type} -> {descendant_type}")
                if direction == "down":
                    response = self.client.scroll_down_by_ancestor_and_descendant(ancestor_type, descendant_type)
                else:
                    response = self.client.scroll_up_by_ancestor_and_descendant(ancestor_type, descendant_type)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using ancestor-descendant")
                    return True
            except Exception as e:
                print(f"Error with ancestor-descendant scroll {direction}: {e}")

        print(f"Failed to scroll {direction} for widget with unique_id: {unique_id}")
        return False

    def scroll_up_or_down_extended(self, all_nodes, unique_id, direction="down", dx=0, dy=100, duration_microseconds=300000, frequency=60):
        """
        Scroll a widget up or down with extended parameters by trying different methods in order of expected reliability.

        Args:
            all_nodes: List of AppNode objects from the widget tree
            unique_id: The unique identifier of the widget to scroll
            direction: "up" or "down" scroll direction
            dx: Horizontal scroll amount (positive = right, negative = left)
            dy: Vertical scroll amount (positive = down, negative = up)
            duration_microseconds: Duration of the scroll gesture in microseconds
            frequency: Frequency of scroll events

        Returns:
            Boolean indicating success or failure
        """
        # Adjust dy based on direction if not explicitly set
        if direction == "up" and dy > 0:
            dy = -dy
        elif direction == "down" and dy < 0:
            dy = -dy

        # Find the node with the matching unique_id
        target_node = None
        for node in all_nodes:
            if node.unique_id == unique_id:
                target_node = node
                break

        if not target_node:
            print(f"No widget found with unique_id: {unique_id}")
            return False

        print(f"Attempting to scroll {direction} with extended parameters: {target_node.widget_type}")

        # Try by Flutter key if available
        if target_node.key:
            try:
                print(f"Trying scroll {direction} by key with extended parameters: {target_node.key}")
                if direction == "down":
                    response = self.client.scroll_down_by_key_extended(target_node.key, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                else:
                    response = self.client.scroll_up_by_key_extended(target_node.key, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using key with extended parameters")
                    return True
            except Exception as e:
                print(f"Error scrolling {direction} by key with extended parameters: {e}")

        # Try by text content if available
        if target_node.text:
            try:
                print(f"Trying scroll {direction} by text with extended parameters: '{target_node.text}'")
                if direction == "down":
                    response = self.client.scroll_down_by_text_extended(target_node.text, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                else:
                    response = self.client.scroll_up_by_text_extended(target_node.text, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using text with extended parameters")
                    return True
            except Exception as e:
                print(f"Error scrolling {direction} by text with extended parameters: {e}")

        # Try by widget type
        try:
            print(f"Trying scroll {direction} by type with extended parameters: {target_node.widget_type}")
            if direction == "down":
                response = self.client.scroll_down_by_type_extended(target_node.widget_type, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
            else:
                response = self.client.scroll_up_by_type_extended(target_node.widget_type, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
            if hasattr(response, 'success') and response.success:
                print(f"Successfully scrolled {direction} using type with extended parameters")
                return True
        except Exception as e:
            print(f"Error scrolling {direction} by type with extended parameters: {e}")

        # Try by tooltip if available
        if 'tooltip' in target_node.properties:
            try:
                tooltip = target_node.properties['tooltip']
                print(f"Trying scroll {direction} by tooltip with extended parameters: {tooltip}")
                if direction == "down":
                    response = self.client.scroll_down_by_tooltip_extended(tooltip, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                else:
                    response = self.client.scroll_up_by_tooltip_extended(tooltip, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using tooltip with extended parameters")
                    return True
            except Exception as e:
                print(f"Error scrolling {direction} by tooltip with extended parameters: {e}")

        # Try with ancestor-descendant approach
        if target_node.parent_node:
            # Get ancestor type (parent widget type)
            ancestor_type = target_node.parent_node.widget_type.split(' ')[0]  # Get main type without description
            # Get descendant type (current widget type)
            descendant_type = target_node.widget_type.split(' ')[0]  # Get main type without description

            try:
                print(f"Trying scroll {direction} by ancestor-descendant with extended parameters: {ancestor_type} -> {descendant_type}")
                if direction == "down":
                    response = self.client.scroll_down_by_ancestor_and_descendant_extended(ancestor_type, descendant_type, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                else:
                    response = self.client.scroll_up_by_ancestor_and_descendant_extended(ancestor_type, descendant_type, dx=dx, dy=dy, duration_microseconds=duration_microseconds, frequency=frequency)
                if hasattr(response, 'success') and response.success:
                    print(f"Successfully scrolled {direction} using ancestor-descendant with extended parameters")
                    return True
            except Exception as e:
                print(f"Error with ancestor-descendant scroll {direction} with extended parameters: {e}")

        print(f"Failed to scroll {direction} with extended parameters for widget with unique_id: {unique_id}")
        return False

    def find_ancestor_with_scroll(self, node):
        """
        Find the closest ancestor of a node that is likely to be scrollable.
        
        Args:
            node: The node to find a scrollable ancestor for
            
        Returns:
            The closest scrollable ancestor node or None if none found
        """
        if not node or not node.parent_node:
            return None
            
        # Define types that are commonly scrollable
        scrollable_types = [
            'scrollview', 'listview', 'gridview', 'pageview', 'singlechildscrollview',
            'customscrollview', 'nestedscrollview', 'refreshindicator', 'scrollable',
            'tabbarview', 'reorderablelistview', 'draggablescrollablesheet',
            'scrollconfiguration', 'scrollphysics', 'scrollbar', 'scrollcontroller',
            'scrollnotification', 'scrollable', 'overscroll'
        ]
        
        # Start with the immediate parent
        current = node.parent_node
        
        # Go up the tree looking for scrollable ancestors
        while current:
            # Check if the current node is likely scrollable
            current_type = current.widget_type.lower()
            if any(scroll_type in current_type for scroll_type in scrollable_types):
                return current
                
            # Check for scrollable properties in the node
            if any(prop in current.properties for prop in ['scroll', 'overflow', 'scrollable']):
                return current
                
            # Move up to the parent
            current = current.parent_node
            
        # No scrollable ancestor found
        return None
        
    def find_descendant_with_scroll(self, node):
        """
        Find the first descendant of a node that is likely to be scrollable.
        
        Args:
            node: The node to find a scrollable descendant for
            
        Returns:
            The first scrollable descendant node or None if none found
        """
        if not node or not node.child_nodes:
            return None
            
        # Define types that are commonly scrollable
        scrollable_types = [
            'scrollview', 'listview', 'gridview', 'pageview', 'singlechildscrollview',
            'customscrollview', 'nestedscrollview', 'refreshindicator', 'scrollable',
            'tabbarview', 'reorderablelistview', 'draggablescrollablesheet',
            'scrollconfiguration', 'scrollphysics', 'scrollbar', 'scrollcontroller',
            'scrollnotification', 'scrollable', 'overscroll'
        ]
        
        # Helper function for DFS traversal
        def find_scrollable_dfs(current):
            # Check if the current node is likely scrollable
            current_type = current.widget_type.lower()
            if any(scroll_type in current_type for scroll_type in scrollable_types):
                return current
                
            # Check for scrollable properties in the node
            if any(prop in current.properties for prop in ['scroll', 'overflow', 'scrollable']):
                return current
                
            # Check children recursively
            for child in current.child_nodes:
                result = find_scrollable_dfs(child)
                if result:
                    return result
                    
            return None
            
        # Start search from the node's children
        return find_scrollable_dfs(node)

    def close(self):
        """Close client and stop service manager if managed by this class"""
        if self._manage_client and hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except Exception:
                pass
        if hasattr(self, 'service_manager'):
            try:
                self.service_manager.stop()
            except Exception:
                pass
                
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        """Destructor to ensure client and service are closed"""
        try:
            self.close()
        except:
            pass