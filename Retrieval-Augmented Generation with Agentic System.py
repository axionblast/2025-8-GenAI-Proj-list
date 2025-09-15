
!python3 -m pip install --no-cache-dir llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
!python3 -m pip install googlesearch-python bs4 charset-normalizer requests-html lxml_html_clean

from pathlib import Path
if not Path('./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf').exists():
    !wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
if not Path('./public.txt').exists():
    !wget https://www.csie.ntu.edu.tw/~ulin/public.txt
if not Path('./private.txt').exists():
    !wget https://www.csie.ntu.edu.tw/~ulin/private.txt


import torch
if not torch.cuda.is_available():
    raise Exception('You are not using the GPU runtime. Change it first or you will suffer from the super slow inference speed!')
else:
    print('You are good to go!')


# Load GPU
# "./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"：Local LLaMA 3.1 8B model file (Q8_0 is quantized format).
# n_gpu_layers=-1：Load all layers of the model onto GPU (accelerates inference).
# n_ctx=16384：Maximum input token length (context window size). 16k fits well for 16GB VRAM, but larger values consume more memory.
# verbose=False：Do not print detailed logs during loading.

from llama_cpp import Llama

# Load the model onto GPU
llama3 = Llama(
    "./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=16384,    # This argument is how many tokens the model can take. The longer the better, but it will consume more memory. 16384 is a proper value for a GPU with 16GB VRAM.
)


# Define the function for generating responses
#_model.create_chat_completion：Call the model for chat-style inference.
#_messages：Input conversation messages (typically a list of system/user messages).
#stop：Stop generation when encountering these tokens.
#max_tokens=512：Maximum number of tokens to generate.
#temperature=0：No randomness (same input → same output). Higher values produce more creativity but may be less stable.
#repeat_penalty=2.0：Penalize repetition. Higher values reduce repeated outputs.


def generate_response(_model: Llama, _messages: str) -> str:
    '''
    This function will inference the model with given messages.
    '''
    _output = _model.create_chat_completion(
        _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=512,    # This argument is how many tokens the model can generate, you can change it and observe the differences.
        temperature=0,      # This argument is the randomness of the model. 0 means no randomness. You will get the same result with the same input every time. You can try to set it to different values.
        repeat_penalty=2.0,
    )["choices"][0]["message"]["content"]
    return _output



# google search tool An async utility for Google search + webpage scraping + plain-text extraction. Input a keyword, it automatically returns plain text from the top search results.

from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
import asyncio
from requests_html import AsyncHTMLSession
import urllib3
urllib3.disable_warnings()

#2 worker function：fetch content of a single webpage

async def worker(s:AsyncHTMLSession, url:str):
    try:
        header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
        if 'text/html' not in header_response.headers.get('Content-Type', ''):
            return None
        r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None

#3. get_htmls function: batch fetch webpages

async def get_htmls(urls):
    session = AsyncHTMLSession()
    tasks = (worker(session, url) for url in urls)
    return await asyncio.gather（*tasks）

#4. search function: search and return webpage plain text
async def search(keyword: str, n_results: int=3) -> List[str]:
    keyword = keyword[:100]  # 限制关键词长度

    #  ① Google search keywords, take 2× results to prevent invalid pages
    results = list(_search(keyword, n_results * 2, lang="zh", unique=True))

    # ② Fetch webpage HTML
    results = await get_htmls(results)

    # ③ Filter out None
    results = [x for x in results if x is not None]

    # ④ Parse HTML using BeautifulSoup
    results = [BeautifulSoup(x, 'html.parser') for x in results]

    # ⑤ Extract plain text (strip whitespace), keep only UTF-8 encoded content
    results = [
        ''.join(x.get_text().split())
        for x in results
        if detect(x.encode()).get('encoding') == 'utf-8'
    ]

    # ⑥ Return top n_results plain text entries
    return results[:n_results]


# testing llm inference pipeline

# You can try out different questions here.
test_question='Who is Taylor Swift？'

messages = [
    {"role": "system", "content": "You are LLaMA-3.1-8B, an AI for answering questions. When using Chinese, always reply in English."},    # System prompt
    {"role": "user", "content": test_question}, # User prompt
]

print(generate_response(llama3, messages))


#agent

class LLMAgent():
    def __init__(self, role_description: str, task_description: str, llm:str="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.role_description = role_description   # Role means who this agent should act like. e.g. the history expert, the manager......
        self.task_description = task_description    # Task description instructs what task should this agent solve.
        self.llm = llm  # LLM indicates which LLM backend this agent is using.
    def inference(self, message:str) -> str:
        if self.llm == 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF': # If using the default one.
            # TODO: Design the system prompt and user prompt here.
            # Format the messsages first.
            messages = [
                {"role": "system", "content": f"{self.role_description}"},  # Hint: you may want the agents to speak Traditional Chinese only.
                {"role": "user", "content": f"{self.task_description}\n{message}"}, # Hint: you may want the agents to clearly distinguish the task descriptions and the user messages. A proper seperation text rather than a simple line break is recommended.
            ]
            return generate_response(llama3, messages)
        else:
            # TODO: If you want to use LLMs other than the given one, please implement the inference part on your own.
            return ""


#TODO
# TODO: Design the role and task description for each agent.

# This agent may help you filter out the irrelevant parts in question descriptions.
question_extraction_agent = LLMAgent(
    role_description="You are a question extraction assistant",
    task_description="Your task is to extract which sentence in the passage is a question?",
)

# This agent may help you extract the keywords in a question so that the search tool can find more accurate results.
keyword_extraction_agent = LLMAgent(
    role_description="You are a keyword extraction assistant",
    task_description="Your task is to extract the key keywords from the passage. For example, in the question 'In which year did Da S pass away?', the keywords are 'Da S', 'pass away', 'which year'. ",
)

# This agent is the core component that answers the question.
qa_agent = LLMAgent(
    role_description="You are LLaMA-3.1-8B, an AI for answering questions. When using Chinese, only respond in Traditional Chinese. You may use RAG research. If you cannot find the answer, do not make things up—just say you don't know.",
    task_description="Please answer the following question:",
)


#RAG pipeline
async def pipeline(question: str) -> str:
    # TODO: Implement your pipeline.
    # Currently, it only feeds the question directly to the LLM.
    # You may want to get the final results through multiple inferences.
    # Just a quick reminder, make sure your input length is within the limit of the model context window (16384 tokens), you may want to truncate some excessive texts.
    return qa_agent.inference(question)

#

from pathlib import Path

# Fill in your student ID first.
STUDENT_ID = "123"

STUDENT_ID = STUDENT_ID.lower()
with open('./public.txt', 'r') as input_f:
    questions = input_f.readlines()
    questions = [l.strip().split(',')[0] for l in questions]
    for id, question in enumerate(questions, 1):
        if Path(f"./{STUDENT_ID}_{id}.txt").exists():
            continue
        answer = await pipeline(question)
        answer = answer.replace('\n',' ')
        print(id, answer)
        with open(f'./{STUDENT_ID}_{id}.txt', 'w') as output_f:
            print(answer, file=output_f)


#testing：

question = "Today is such a nice day. On this beautiful Sunday morning, I had a cup of coffee. At noon I planned to play games with friends, we first played Overwatch 2, then ate dumplings. In the afternoon we played Valorant. I am in New Jersey, USA. May I ask: who is the publisher/operator of the second game I played?"  
answer = await pipeline(question)      # pipeline async
print(answer)

# answer：

# According to your description, the second game you played is Valorant. In mainland China, Valorant is operated by Tencent. However, since you mentioned New Jersey, USA, that does not fall under the China region.
# Therefore, the answer may differ because different countries or regions may have different publishers. But based on general information, we know:
# * In the United States and Canada, Valorant is operated directly by Riot Games.
# Therefore, in New Jersey, USA, it is directly operated by Riot Games.
