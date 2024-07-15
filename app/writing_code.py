from agents.software_dev_agents import client_user_proxy_agent, cto_assistant_agent

task = """
Write python code to output numbers 1 to 100, and then store the code in a file
"""

client_user_proxy_agent.initiate_chat(
    cto_assistant_agent,
    message=task
)

task2 = """
Change the code in the file you just created to instead output numbers 1 to 200
"""

client_user_proxy_agent.initiate_chat(
    cto_assistant_agent,
    message=task2
)