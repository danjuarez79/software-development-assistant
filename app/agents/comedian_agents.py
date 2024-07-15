import autogen

# import sys
# import pathlib
# path = pathlib.Path().absolute().parent.as_posix()+"/software-development-assistant"
# sys.path.append(path)
import config.llm_config_lists as llm_config_lists

cathy_llm_config = {
    # "request_timeout": "600",
    "cache_seed": 1,
    "temperature": 0.9,
    "max_tokens": 1000,
    "config_list": llm_config_lists.lmstudio_config_list
}
joe_llm_config = {
    # "request_timeout": "600",
    "cache_seed": 2,
    "temperature": 0.7,
    "max_tokens": 1000,
    "config_list": llm_config_lists.lmstudio_config_list
}

cathy = autogen.ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians. You expect jokes to be very funny an inelligent jokes. You are always critical with others.",
    llm_config=cathy_llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)

joe = autogen.ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians. You are a creative comedian who loves to joke about star trek game app and don't like simple jokes.",
    llm_config=joe_llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)