import autogen

from config import llm_config_lists

llm_config={
    "timeout": 600,
    "cache_seed": 42,
    "config_list": llm_config_lists.lmstudio_config_list,
    "temperature": 0
}

cto_assistant_agent = autogen.AssistantAgent(
    name="CTO",
    llm_config=llm_config,
    system_message="Chief technical officer of a tech company"
)

client_user_proxy_agent = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web","use_docker": False},
    llm_config=llm_config,
    system_message="""Reply with the text TERMINATE if the task has been solved at full satisfaction and don't express any gratitude.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

