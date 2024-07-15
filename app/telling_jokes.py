import sys
import pathlib
# agents_path = pathlib.Path().absolute().parent.as_posix()+"/software-development-assistant/agents"
# sys.path.append(agents_path)
# root_path = pathlib.Path().absolute().parent.as_posix()+"/software-development-assistant/app"
# sys.path.append(root_path)
import agents.comedian_agents as COMEDIAN

import logging
class CostWarningFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Model") or "is not found. The cost will be 0" not in record.getMessage()

logging.getLogger("autogen.oai.client").addFilter(CostWarningFilter())

result = COMEDIAN.joe.initiate_chat(COMEDIAN.cathy, message="Cathy, tell me a joke unique joke.", max_turns=2)

# if __name__ == '__main__':
    # import sys
    # import pathlib
    # agents_path = pathlib.Path().absolute().parent.as_posix()+"/software-development-assistant/agents"
    # root_path = pathlib.Path().absolute().parent.as_posix()+"/software-development-assistant"
    # sys.path.append(agents_path)
    # sys.path.append(root_path)