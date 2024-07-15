import os
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chains import TransformChain, SequentialChain

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

load_dotenv()
# openai_api_key = os.getenv('OPENAI_API_KEY')
# client = OpenAI(api_key=openai_api_key)
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

current_memo_path = "langchain_app/chat_history.txt"
full_memo_path = "langchain_app/Conversation.txt"

user_response = input("Would you like to continue your previous conversation? (yes/no): ")
if user_response.lower() == 'no':
    # we need to reset both files
    with open(full_memo_path, 'w') as file:
        file.write("[0]\n\n")

    with open(current_memo_path, 'w') as file:
        file.write("")

# Ok, this is where I'll write the part dealing with the last exchange
# this part is only supposed to return the list with the number of all the exchanges in Conversation.txt
exchange_numbers = []
exchanges_text = []
with open(full_memo_path, 'r') as file:
    j = -1
    for line in file:
        if line.startswith('['):
            number = int(line.strip()[1:-1])
            exchange_numbers.append(number)
            j = j + 1
            exchanges_text.append('')
        if line.strip():
            line = re.sub(r'(\r\n|\r|\n)+', ' ', line)  # Replace newlines with spaces
            exchanges_text[j] += line
# it works, the list is well done
# I'll just need to set len(exchange_numbers) at the end
# but here I'm getting the three last exchanges: (slicing is flexible so no need to account for first iterations)
last_elements = ' '.join(exchanges_text[-3:])

persist_directoryT = 'langchain_app/persist_chroma'
embedding_f = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})
vectorstore = Chroma(persist_directory=persist_directoryT, embedding_function=embedding_f)

chat_model = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    base_url="http://localhost:1234/v1",
    temperature=0.3
)

# sys_prompt: PromptTemplate = PromptTemplate(
#     input_variables=["current_chat_history", "user_query"],
#     template="""**System Prompt** Your role is to reformulate (if needed) the queries based solely on the information 
#     provided to you. That information will start with **Start of Context** and ends at **End of Context**. 
#     The rest is the prompt or query.

#     It is essential to rely strictly on this context, without drawing on any external knowledge.
#     Adhere to the following guidelines when crafting your responses:

#     1. **Stand alone Query**: If the query is specific and NOT related to the information within 
#     your provided context, Do NOT reformulate the initial query (e.g: "Who is Stalin?" if the chat history so far only
#     talked about cats, do not change the query). Output the user's initial query!

#     2. **General or Ambiguous Queries**: If the query is general, ambiguous, or seems to imply a need for context 
#     (e.g., "Could you provide more details about that?" without specifying "that"), 
#     acknowledge this explicitly. Use the context to make sense of the question and reformulate it so that anyone could
#     understand it even without context.
#     The same is true if the user mentions a past conversation or a previous exchange. 
#     YOU DO NOT HAVE TO REFORMULATE if you are unsure. 
#     DO NOT TRY to answer the question under any circumstances, no matter the question.
#     Do not waste context with useless information.

#     3. **No chat history**: If you do not have enough information in the provided chat history, output the original query.

#     This refined approach ensures you effectively communicate the limitations of the provided context and 
#     guide users to where they might find a more complete answer, thus improving the user experience 
#     by setting clear expectations. **End of System Prompt**
#     """)
sys_prompt: PromptTemplate = PromptTemplate(
    input_variables=["current_chat_history", "user_query"],
    template="""
    """)
system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

student_prompt: PromptTemplate = PromptTemplate(
    input_variables=["current_chat_history", "user_query"],
    template="Using only {current_chat_history}, answer {user_query}. Follow the Guidelines!")

student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, student_message_prompt])


def transform_func(inputs: dict) -> dict:
    documents = inputs["current_chat_history"]
    n = len(documents)
    cleaned_texts = "**Start of Context**"
    # I might add new things to delete later to improve the cleaning process.
    i = 0
    for document, score in documents:
        page_content = document.page_content
        page_content = re.sub(r'(\r\n|\r|\n)+', ' ', page_content)  # Replace newlines with spaces
        page_content = re.sub(r'[ \t]+', ' ', page_content)  # Collapse multiple spaces/tabs into one
        if i == n:
            cleaned_texts += page_content + "**End of Context**"
        else:
            cleaned_texts += page_content + " "
    cleaned_texts = cleaned_texts.strip()
    return {"output_text": cleaned_texts}


clean_extra_spaces_chain = TransformChain(
    input_variables=["current_chat_history"],
    output_variables=["output_text"],
    transform=transform_func)

style_paraphrase_chain = LLMChain(llm=chat_model, prompt=chat_prompt,
                                  output_key='final_output')  # c'est la désignation de la sortie de la première chaine

# I wonder if I can put two SequentialChains together inside another SequentialChain
"""seq_chain = SequentialChain(
    chains=[clean_extra_spaces_chain, style_paraphrase_chain],
    input_variables=['current_chat_history', 'user_query'],
    output_variables=['final_output'])"""
# And that's it for the 'retrieval' chain. Below starts the main chat chain

# I'll have to format the provided context as well as both histories.
sys_prompt2: PromptTemplate = PromptTemplate(
    input_variables=["final_output", "original_sentence"],
    template="""**System Prompt** Your role is to respond to queries based on the information provided 
    to you. There will be two sources of info: the chat history, delimited by **Chat history** and **/Chat history** 
    and the context indicated by **Start Context** and **End Context**. DO NOT use your own knowledge.
    The rest is the prompt or query.

    It is essential to rely strictly on both context and chat history, without drawing on any external knowledge.
    Adhere to the following guidelines when crafting your responses:

    1. **Conflict**: If the context and the chat history are in disagreement, focus on the chat history.
    for instance if the context say that you do not have 'something', but that 'something' is in your
    chat history, then rely on the chat history for answer

    2. **History Queries**: The query you receive comes from another model.
    Therefore you might get a confusing question. If the user asks a general, ambiguous, or seems to 
    imply a need for context beyond what is provided 
    (e.g., "Could you provide more details about that?" without specifying "that"), you need to understand that the
    user refers to the latest topic you discussed with him. In such cases where the context provided doesn't
    help answering the question, focus on using your chat history, and invite the user to ask a more specific
    question.
    It is very important that you use your chat history as much as the context if not more. If the provided context is
    not helpful, focus entirely on your chat history and push the user to ask another more specific question.

    3. **Wrong Context** If you consider that the context provided does not answer the question at all
    (e.g, the user asks about dolphin and the context is about school), ignore the context. Do not provide
    information that is not in your chat history nor context. It is fine NOT to answer in those case.

    This refined approach ensures you effectively communicate the limitations of the provided context and 
    guide users to where they might find a more complete answer, thus improving the user experience 
    by setting clear expectations. **End of System Prompt**
    """)
system_prompt2 = SystemMessagePromptTemplate(prompt=sys_prompt2)

# it might be useful to decompose current_chat_history into current_chat_history and latest_exchanges
student_prompt2: PromptTemplate = PromptTemplate(
    input_variables=["final_output", "original_sentence"],
    template="""Using only **Start Context** {original_sentence} **End Context**,
    answer {final_output}. Follow the Guidelines!""")

student_prompt_2 = HumanMessagePromptTemplate(prompt=student_prompt2)
chat_prompt2 = ChatPromptTemplate.from_messages(
    [system_prompt2, student_prompt_2])

main_chain = LLMChain(llm=chat_model, prompt=chat_prompt2, output_key='user_output')

# creation of the chain in charge of memory:
sys_promptSM: PromptTemplate = PromptTemplate(
    input_variables=["user_output", "user_query", "current_chat_history"],
    template="""**System Prompt** Your role is to use both the current chat history, and the latest exchange
    between the user and the model to create a summary to save the context. The summary you will create will replace
    the current chat history and will be used by another model, so it needs to only keep the important,
    non-redundant information.

    It is essential to rely strictly on both context and chat history, without drawing on any external knowledge.
    Adhere to the following guidelines when crafting your responses:

    1. **Named Entities**: If the user mentions his name/ identity, you should make sure that the summary always include
    it. It is crucial that the important named entities stay in memory.

    2. **Topics**: Pay attention to the user's responses to decide what information to keep in the summary.
    If the user is dissatisfied with the model's response, then there is no need to include much of it.
    If the user decides to switch topics, then focus on keeping the latest one is you are near the 2500 words limit.
    If the summaries for both topics combined won't use all of the available memory, then keep both.

    3. **Brevity and Uniqueness** Try to keep as much relevant information as possible in your summaries, however, 
    your context window is finite, so keep the summaries you do UNDER 2500 words.

    4. **Self-Assessment for Relevance**: After drafting a summary, briefly assess its content for 
    redundancy and relevance. Ensure that the summary reflects the latest developments in the conversation, 
    removing or condensing any repetitive information unless it is essential for clarity or continuity.

    5. **Avoid Repetition**: Vigilantly avoid repeating information in your summaries. 
    Repetition should only be employed for emphasis on newly established critical points or when summarizing 
    complex concepts that require reiteration for clarity. Always aim to introduce new information or perspectives 
    in the summary.

    This approach ensures you effectively improves the user experience 
    by allowing the main model to keep giving relevant answers. **End of System Prompt**
    """)
system_promptSM = SystemMessagePromptTemplate(prompt=sys_promptSM)

student_promptSM: PromptTemplate = PromptTemplate(
    input_variables=["user_output", "user_query", "current_chat_history"],
    template="""Using {current_chat_history}, {user_query} and {user_output} create a new summary.
             Follow the Guidelines!""")
student_prompt_SM = HumanMessagePromptTemplate(prompt=student_promptSM)
chat_prompt2 = ChatPromptTemplate.from_messages(
    [system_promptSM, student_prompt_SM])

memory_summary_chain = LLMChain(llm=chat_model, prompt=chat_prompt2, output_key='new_chat_history')

seq_chain2 = SequentialChain(
    chains=[main_chain, memory_summary_chain],
    input_variables=["final_output", "user_query", "original_sentence", "current_chat_history"],
    output_variables=['user_output', 'new_chat_history'])

query = input("Ask anything!: ")

with open(current_memo_path, 'r') as file:
    chat_history = file.read()

chat_history = chat_history + last_elements

result = style_paraphrase_chain.invoke({'current_chat_history': chat_history, 'user_query': query})
# print("Retriever's reponse: \n")
print(result['final_output'])

# Using retrieval chain's results to test the main chain.
input2 = result['final_output']

test_text = vectorstore.similarity_search_with_score(input2, 8)
tests2 = seq_chain2.invoke({
    'final_output': input2,
    'user_query': query,
    'original_sentence': test_text,
    'current_chat_history': chat_history})

print('\nAnswer: ')
response = tests2['user_output']
print(response)
# print('\nNew Chat History: ')
new_chat_history = tests2['new_chat_history']
# print(new_chat_history)

with open(current_memo_path, 'w') as file:
    file.write(new_chat_history)

# This is the part that write the exchanges, but it needs to now the number of the previous exchange.
with open(full_memo_path, 'a') as file:
    num = str(len(exchange_numbers))
    exchange = '[' + num + ']\n(user question): ' + query + '\n(model response): ' + response + '\n\n'
    file.write(exchange)
