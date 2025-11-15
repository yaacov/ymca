"""MCP client for connecting to MCP servers and using their tools."""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for connecting to MCP (Model Context Protocol) servers.
    
    MCP servers are external processes that provide tools to AI models.
    This client handles:
    - Starting MCP server processes
    - Discovering available tools
    - Executing tool calls via JSON-RPC
    """
    
    def __init__(self, name: str, command: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize MCP client.
        
        Args:
            name: Server name (for identification)
            command: Command and arguments to start the server
            env: Optional environment variables for the server
        """
        self.name = name
        self.command = command
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self.tools: Dict[str, Dict] = {}
        self.next_id = 1
        
        logger.info(f"Initialized MCP client for '{name}'")
    
    def start(self):
        """Start the MCP server process."""
        try:
            logger.info(f"Starting MCP server '{self.name}': {' '.join(self.command)}")
            
            # Start the server process (stderr goes to our stderr so we can see errors)
            import os
            env = os.environ.copy()
            env.update(self.env)
            
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr
                text=True,
                bufsize=1,
                env=env
            )
            
            # Initialize the connection
            logger.debug(f"Sending initialize request to MCP server '{self.name}'")
            init_response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ymca-chat",
                    "version": "1.0.0"
                }
            })
            
            if not init_response or "error" in init_response:
                error_msg = init_response.get("error", {}).get("message", "Unknown error") if init_response else "No response"
                raise RuntimeError(f"MCP server '{self.name}' initialization failed: {error_msg}")
            
            logger.debug(f"MCP server '{self.name}' initialized successfully")
            
            # List available tools
            logger.debug(f"Requesting tools list from MCP server '{self.name}'")
            response = self._send_request("tools/list", {})
            
            if not response:
                raise RuntimeError(f"MCP server '{self.name}' returned no response for tools/list")
            
            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                raise RuntimeError(f"MCP server '{self.name}' tools/list failed: {error_msg}")
            
            if response and "result" in response and "tools" in response["result"]:
                for tool in response["result"]["tools"]:
                    tool_name = tool["name"]
                    self.tools[tool_name] = {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                    }
                    logger.debug(f"  Discovered tool: {tool_name}")
                
                logger.info(f"MCP server '{self.name}' ready with {len(self.tools)} tools")
            else:
                logger.warning(f"MCP server '{self.name}' returned no tools")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.name}': {e}")
            # Try to read stderr to see if there are any error messages
            if self.process and self.process.stderr:
                try:
                    import select
                    # Non-blocking read of stderr
                    if select.select([self.process.stderr], [], [], 0)[0]:
                        stderr_output = self.process.stderr.read()
                        if stderr_output:
                            logger.error(f"MCP server stderr: {stderr_output}")
                except:
                    pass
            raise
    
    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info(f"MCP server '{self.name}' stopped")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning(f"MCP server '{self.name}' force killed")
            except Exception as e:
                logger.error(f"Error stopping MCP server '{self.name}': {e}")
    
    def get_tools(self) -> Dict[str, Dict]:
        """
        Get available tools from the server.
        
        Returns:
            Dictionary of tool name -> tool definition
        """
        return self.tools.copy()
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            ValueError: If tool not found or call fails
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in MCP server '{self.name}'")
        
        logger.info(f"Calling MCP tool: {self.name}.{tool_name}")
        logger.debug(f"  Arguments: {arguments}")
        
        try:
            response = self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            
            if response and "result" in response:
                result = response["result"]
                
                # Extract content from MCP response
                if "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Return first content item's text
                        return content[0].get("text", str(content))
                    return str(content)
                
                return result
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                raise ValueError(f"MCP tool call failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            raise
    
    def _send_request(self, method: str, params: Dict) -> Optional[Dict]:
        """
        Send JSON-RPC request to MCP server.
        
        Args:
            method: RPC method name
            params: Method parameters
            
        Returns:
            Response dictionary or None on error
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError(f"MCP server '{self.name}' not running")
        
        request_id = self.next_id
        self.next_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            logger.debug(f"MCP request: {method}")
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                logger.error(f"MCP server '{self.name}' closed connection")
                return None
            
            response = json.loads(response_line)
            logger.debug(f"MCP response: {response.get('id', 'unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in MCP communication: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

