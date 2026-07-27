from datetime import datetime

def tool_calls(messages):
    #print("did agent use tool?")
    for msg in messages:
        # AI Tool Call
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            #print("agent did use tool!")
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "UnknownTool")
                    tool_args = tool_call.get("args", {})
                else:
                    tool_name = getattr(tool_call, "name", "UnknownTool")
                    tool_args = getattr(tool_call, "args", {})

                agent_step(
                    event="Tool selected",
                    action=str(tool_name),
                    action_input=str(tool_args),
                    output="Pending"
                )

        # Tool Response
        if msg.__class__.__name__ == "ToolMessage":            
            tool_name = getattr(msg, "name", "UnknownTool")
            tool_output = getattr(msg, "content", "")

            agent_step(
                event="Tool response",
                action=str(tool_name),
                action_input="N/A",
                output=str(tool_output)
            )

def agent_step(
    event: str,
    action: str,
    action_input: str,
    output: str
):
    """
    Save agent trajectory logs.
    """

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_text = (
        f"[{timestamp}] "
        f"EVENT: {event} | "
        f"ACTION: {action} | "
        f"INPUT: {action_input} | "
        f"OUTPUT: {str(output)[:300]}\n\n"
    )

    with open(
        "states/agent_trajectory.txt",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(log_text)
