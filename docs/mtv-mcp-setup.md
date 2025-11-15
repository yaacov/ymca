# MTV MCP Server Configuration

This guide explains how to configure YMCA to work with the MTV (Migration Toolkit for Virtualization) MCP server for VM migration workflows.

## Overview

The MTV MCP server provides tools for migrating virtual machines from vSphere, oVirt, OpenStack, and OVA sources to Kubernetes/OpenShift using kubectl-mtv and Forklift. By configuring YMCA with the appropriate system prompt and MCP server flags, you can create a specialized assistant for VM migration tasks.

## MCP Server Setup

### Installing the MTV MCP Server

Install in your virtual environment:

```bash
# Ensure your virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install kubectl-mtv with MCP server support
pip install mcp-server-kubectl-mtv
```

### Starting YMCA with MTV MCP Server

Use the `--mcp-server` flag to connect to the MTV MCP server:

```bash
# Chat interface with MTV MCP server
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server"

# Web interface with MTV MCP server
ymca-web --mcp-server "mtv:kubectl-mtv mcp-server"
```

The format is `<namespace>:<command>`, where:
- `namespace` (mtv) is the prefix used for tool calls (e.g., `mtv.CreateProvider`)
- `command` (kubectl-mtv mcp-server) is the command to start the MCP server

## System Prompt Configuration

To get the best results with the MTV MCP server, configure a specialized system prompt. YMCA includes a pre-configured MTV system prompt optimized for VM migration workflows.

### Method 1: Using --system-prompt Flag with @filename

The easiest way is to use the included MTV system prompt file:

```bash
# Chat interface with MTV system prompt
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

# Web interface with MTV system prompt
ymca-web --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt
```

The `@` prefix tells YMCA to load the prompt from a file. You can also use an absolute path:

```bash
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @/path/to/your-custom-prompt.txt
```

### Quick Start

Run YMCA with MTV configuration:

```bash
# Start CLI chat with MTV configuration
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

# Start web interface with MTV configuration
ymca-web --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

# With custom host/port
ymca-web --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt --host 0.0.0.0 --port 9000
```

### Method 2: Programmatic Configuration

When using YMCA as a library, configure the system prompt when initializing the ChatAPI:

```python
from ymca import ChatAPI, ModelHandler, MemoryTool

model_handler = ModelHandler(model_path="path/to/model.gguf")
memory_tool = MemoryTool(memory_dir="./data/tools/memory")

system_message = """Your MTV system prompt here..."""

chat_api = ChatAPI(
    model_handler=model_handler,
    memory_tool=memory_tool,
    system_message=system_message,
    mcp_servers={"mtv": "kubectl-mtv mcp-server"}
)
```

## MTV System Prompt

YMCA includes a pre-configured system prompt optimized for MTV operations in `docs/mtv-system-prompt.txt`. This prompt:

- Instructs the assistant to check cluster state first using MTV tools
- Prioritizes live cluster data over documentation
- Provides clear tool selection strategy
- Enforces strict no-hallucination policy
- Optimizes for concise, actionable responses

The prompt is designed to work seamlessly with both the MTV MCP server and the memory tool for documentation retrieval.

## Available MTV Tools

The MTV MCP server provides tools for:

### Provider Management
- `mtv.CreateProvider` - Create connections to vSphere, oVirt, OpenStack, OpenShift, or OVA sources
- `mtv.ListResources` - List providers, plans, mappings, hosts, and hooks
- `mtv.PatchProvider` - Update provider configuration
- `mtv.DeleteProvider` - Remove providers

### Inventory Exploration
- `mtv.ListInventory` - Query VMs, networks, storage, hosts, and other resources
- Supports complex queries using Tree Search Language (TSL)

### Mapping Management
- `mtv.ManageMapping` - Create, patch, or delete network and storage mappings
- Automatic validation for network and storage constraints

### Plan Management
- `mtv.CreatePlan` - Create migration plans with VM selection and configuration
- `mtv.PatchPlan` - Modify plan settings
- `mtv.PatchPlanVm` - Customize individual VM settings
- `mtv.GetPlanVms` - Monitor VM migration status

### Plan Lifecycle
- `mtv.ManagePlanLifecycle` - Start, cancel, cutover, archive, or unarchive plans

### Advanced Features
- `mtv.CreateHost` - Configure ESXi hosts for direct data transfer
- `mtv.CreateHook` - Set up pre/post-migration automation
- `mtv.GetLogs` - Retrieve logs for troubleshooting
- `mtv.GetMigrationStorage` - Inspect PVCs and DataVolumes

## Example Usage

### Complete Migration Workflow

```bash
# 1. Start YMCA with MTV MCP server and system prompt
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

# 2. In the chat:
User: "What providers do I have?"
Assistant: [Calls mtv.ListResources to check actual cluster state]

User: "Create a vSphere provider for vcenter.example.com"
Assistant: [Calls mtv.CreateProvider with appropriate parameters]

User: "Show me all VMs that are powered on"
Assistant: [Calls mtv.ListInventory with query "where powerState = 'On'"]

User: "Create a migration plan for VMs web-01 and web-02"
Assistant: [Calls mtv.CreatePlan with selected VMs and auto-creates mappings]

User: "Start the migration"
Assistant: [Calls mtv.ManagePlanLifecycle with action="start"]
```

### Loading MTV Documentation

To enhance the assistant's ability to explain MTV concepts and commands:

```bash
# Load MTV documentation into memory
ymca-memory load-docs /path/to/kubectl-mtv/docs

# The assistant can now use retrieve_memory for documentation questions
```

## Cluster Access Requirements

The MTV MCP server requires:
- Access to a Kubernetes/OpenShift cluster with MTV operator installed
- Valid kubeconfig with appropriate permissions
- Network connectivity to source virtualization platforms (vSphere, oVirt, etc.)

## Troubleshooting

### MCP Server Connection Issues

If the MCP server fails to connect:

1. Verify kubectl-mtv is installed: `kubectl-mtv version`
2. Check cluster access: `kubectl get forklift -A`
3. Test MCP server manually: `kubectl-mtv mcp-server`

### Tool Call Failures

If MTV tools return errors:

1. Check cluster connectivity
2. Verify provider credentials
3. Review logs: Use `mtv.GetLogs` tool
4. Check resource status: Use `mtv.ListResources` with appropriate filters

## Best Practices

1. **Always check cluster state first** - Use MTV tools before making suggestions
2. **Combine tools and memory** - Use MTV tools for cluster state, retrieve_memory for documentation
3. **Validate before actions** - Check providers and inventory before creating plans
4. **Monitor progress** - Use GetPlanVms to track migration status
5. **Handle errors gracefully** - Check tool results and provide actionable guidance

## Additional Resources

- kubectl-mtv documentation: https://github.com/kubev2v/kubectl-mtv
- MTV Operator: https://github.com/kubev2v/forklift
- MCP Protocol: https://modelcontextprotocol.io

