import logging
import json

# Configure logging to write detailed debug information to a file.
logging.basicConfig(
    level=logging.DEBUG,
    filename="agentic.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Define the available tools.
# For demonstration, we define two simple tools: a calculator and an echo function.
AVAILABLE_TOOLS = {
    "calculator": lambda args: f"Calculated result: {sum(args.get('numbers', []))}",
    "echo": lambda args: f"Echo: {args.get('message', '')}"
}

def call_llm(prompt):
    """
    Mock LLM call.

    This function simulates an LLM's output.
    For demonstration, if the prompt contains keywords like "calculate" or "echo",
    the LLM returns a structured tool call using a special marker ("CALL_TOOL:").

    Replace this with your actual LLM API call.
    """
    if "calculate" in prompt.lower():
        # Simulate a tool call request for the calculator.
        tool_call = {
            "tool": "calculator",
            "args": {"numbers": [1, 2, 3]}  # Example: a simple sum operation.
        }
        return "CALL_TOOL:" + json.dumps(tool_call)
    elif "echo" in prompt.lower():
        tool_call = {
            "tool": "echo",
            "args": {"message": "Hello from LLM"}
        }
        return "CALL_TOOL:" + json.dumps(tool_call)
    else:
        # If no tool call is needed, return a final text response.
        return "Final response from LLM: " + prompt

def parse_llm_response(response):
    """
    Parses the LLM response to detect if it contains a tool call.

    We assume that tool call requests are prefixed with "CALL_TOOL:" followed by a JSON payload.
    If such a marker exists, return a tuple (tool_name, args). Otherwise, return None.
    """
    marker = "CALL_TOOL:"
    if response.startswith(marker):
        try:
            tool_data = json.loads(response[len(marker):])
            tool_name = tool_data.get("tool")
            args = tool_data.get("args", {})
            return tool_name, args
        except json.JSONDecodeError as e:
            logging.error("Failed to parse tool call JSON: %s", e)
            return None
    return None

def validate_tool_call(tool_name, args):
    """
    Validates that the requested tool exists in our defined tool set.

    More sophisticated validation on the arguments can be performed as needed.
    """
    return tool_name in AVAILABLE_TOOLS

def execute_tool(tool_name, args):
    """
    Executes the tool if it exists.

    In a real system, this function could perform more complex operations or call out
    to external services.
    """
    try:
        tool_func = AVAILABLE_TOOLS[tool_name]
        return tool_func(args)
    except Exception as e:
        logging.error("Error during tool execution: %s", e)
        return f"Error: {str(e)}"

def format_tool_response(tool_response):
    """
    Formats the tool's output.

    This could involve additional formatting or error handling.
    Here, we simply convert it to a string.
    """
    return str(tool_response)

def update_prompt_with_tool_output(original_response, tool_output):
    """
    Incorporates the tool's output back into the conversation context.

    In this simple example, we concatenate the tool output to the original response.
    In your application, you might structure this update differently.
    """
    updated_prompt = original_response + "\nTool Output: " + tool_output
    return updated_prompt

def agent_loop(user_prompt):
    """
    Main agent loop:

    1. Logs and sends the current prompt to the LLM.
    2. Parses the LLM's output looking for tool call requests.
    3. Validates and executes any tool calls.
    4. Updates the prompt with the tool's output.
    5. Repeats until a final LLM response (without tool call marker) is received.
    """
    current_prompt = user_prompt
    while True:
        logging.info("User Prompt: %s", current_prompt)
        llm_response = call_llm(current_prompt)
        logging.info("LLM Response: %s", llm_response)

        # Check if the LLM response includes a tool call.
        tool_call = parse_llm_response(llm_response)
        if tool_call:
            tool_name, tool_args = tool_call
            logging.info("Detected Tool Call: tool=%s, args=%s", tool_name, tool_args)
            if validate_tool_call(tool_name, tool_args):
                tool_result = execute_tool(tool_name, tool_args)
                formatted_tool_output = format_tool_response(tool_result)
                logging.info("Tool Output: %s", formatted_tool_output)
                # Feed the tool output back into the prompt for further processing.
                current_prompt = update_prompt_with_tool_output(llm_response, formatted_tool_output)
            else:
                error_message = f"Invalid tool call: {tool_name}"
                logging.error(error_message)
                current_prompt = update_prompt_with_tool_output(llm_response, error_message)
        else:
            # No tool call detected means we have the final response.
            logging.info("Final LLM Response: %s", llm_response)
            return llm_response

if __name__ == "__main__":
    # Get the initial prompt from the user.
    initial_prompt = input("Enter your prompt: ")
    final_response = agent_loop(initial_prompt)
    print("Final Response:\n", final_response)
