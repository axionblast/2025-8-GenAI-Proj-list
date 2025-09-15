# 1 Install packages. 
!pip install srt==3.5.3
!pip install datasets==2.20.0
!pip install DateTime==5.5
!pip install OpenCC==1.1.7
!pip install opencv-contrib-python==4.8.0.76
!pip install opencv-python==4.8.0.76
!pip install opencv-python-headless==4.10.0.84
!pip install openpyxl==3.1.4
!pip install openai==1.35.3
!pip install git+https://github.com/openai/whisper.git@ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
!pip install numpy==1.25.2
!pip install soundfile==0.12.1
!pip install -q -U google-generativeai==0.7.0
!pip install anthropic==0.29.0

# Import packages.
import whisper
import srt
import datetime
import time
import os
import re
import pathlib
import textwrap
import numpy as np
import soundfile as sf
from opencc import OpenCC
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import google.generativeai as genai
import anthropic

# Load dataset.
dataset_name = ""
dataset = load_dataset(dataset_name)

# Prepare audio.
input_audio = dataset["test"]["audio"][0]
input_audio_name = input_audio["path"]
input_audio_array = input_audio["array"].astype(np.float32)
sampling_rate = input_audio["sampling_rate"]

print(f"Now, we are going to transcribe the audio: Signals and Life Professor Lin-Shan(2023) ({input_audio_name}).")


# 2 Automatic Speech Recognition (ASR)

# Define speech recognition function
def speech_recognition(model_name, input_audio, output_subtitle_path, decode_options, cache_dir="./"):
    # load model
    model = whisper.load_model(name=model_name, download_root=cache_dir)

    # transcribe audio
    transcription = model.transcribe(
        audio=input_audio,
        language=decode_options["language"],
        verbose=False,
        initial_prompt=decode_options["initial_prompt"],
        temperature=decode_options["temperature"]
    )

    # process results, generate subtitle file
    subtitles = []
    for i, segment in enumerate(transcription["segments"]):
        start_time = datetime.timedelta(seconds=segment["start"])
        end_time = datetime.timedelta(seconds=segment["end"])
        text = segment["text"]
        subtitles.append(srt.Subtitle(index=i, start=start_time, end=end_time, content=text))

    srt_content = srt.compose(subtitles)

    # save file
    with open(output_subtitle_path, "w", encoding="utf-8") as file:
        file.write(srt_content)

    print(f"Saving the subtitle to  {output_subtitle_path}")

# Parameter Setting of Whisper

# @title Parameter Setting of Whisper { run: "auto" }

''' In this block, you can modify your desired parameters and the path of input file. '''

# The name of the model you want to use.
# For example, you can use 'tiny', 'base', 'small', 'medium', 'large-v3' to specify the model name.
# @markdown **model_name**: The name of the model you want to use.
model_name = "medium" # @param ["tiny", "base", "small", "medium", "large-v3"]

# Define the suffix of the output file.
# @markdown **suffix**: The output file name is "output-{suffix}.* ", where .* is the file extention (.txt or .srt)
suffix = "Signals and Life" # @param {type: "string"}

# Path to the output file.
output_subtitle_path = f"./output-{suffix}.srt"

# Path of the output raw text file from the SRT file.
output_raw_text_path = f"./output-{suffix}.txt"

# Path to the directory where the model and dataset will be cached.
cache_dir = "./"

# The language of the lecture video.
# @markdown **language**: The language of the lecture video.
language = "zh" # @param {type:"string"}

# Optional text to provide as a prompt for the first window.
# @markdown **initial_prompt**: Optional text to provide as a prompt for the first window.
initial_prompt = "use traditional chinese" #@param {type:"string"}

# The temperature for sampling from the model. Higher values mean more randomness.
# @markdown  **temperature**: The temperature for sampling from the model. Higher values mean more randomness.
temperature = 0 # @param {type:"slider", min:0, max:1, step:0.1}

# Size (大小)：表示模型的尺寸，不同大小的模型训练时使用的数据量不同，因此性能和精度也不同。较大的模型通常会有更高的精度。Medium 是个不错的选择，tiny 和 base 效果一般，用于学习的话也可以。
# Parameters (参数量)：模型的参数数量，表示模型的复杂度。参数越多，模型的性能通常越好，但也会占用更多的计算资源。
# English-only model (仅限英文模型)：模型的标识名称，只用于处理英文音频转录，适用于仅需要处理英文语音的场景。
# Multilingual model (多语言模型)：模型的标识名称，用于在代码中加载相应的模型，对应于接下来的 model_name 参数。
# Required VRAM (所需显存)：指运行该模型时所需的显存大小。如果你对参数和显存的对应关系感兴趣，可以阅读之前的文章：《07. 探究模型参数与显存的关系以及不同精度造成的影响.md》。
# Relative speed (相对速度)：相对速度表示模型处理语音转录任务的效率。数字越大，模型处理速度越快，与模型的参数量成反比。

# Construct DecodingOptions
decode_options = {
    "language": language,
    "initial_prompt": initial_prompt,
    "temperature": temperature
}

# print message.
message = "Transcribe Signals and Life Professor Lin-Shan(2023)"
print(f"Setting: (1) Model: whisper-{model_name} (2) Language: {language} (2) Initial Prompt: {initial_prompt} (3) Temperature: {temperature}")
print(message)

# Running ASR.
speech_recognition(model_name=model_name, input_audio=input_audio_array, output_subtitle_path=output_subtitle_path, decode_options=decode_options, cache_dir=cache_dir)

#  Read and print subtitle content
with open(output_subtitle_path, 'r', encoding='utf-8') as file:
    content = file.read()
print(content)

# Part3 - Preprocess the results of automatic speech recognition

# Extract subtitle text
def extract_and_save_text(srt_filename, output_filename):
    # Read SRT file
    with open(srt_filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove timestamps and indexes
    pure_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
    pure_text = re.sub(r'\n\n+', '\n', pure_text)

    # Save pure text
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(pure_text)

    print(f'Extracted text has been saved to {output_filename}')

    return pure_text

# Split text
def chunk_text(text, max_length):
    return textwrap.wrap(text, max_length)

# Chunk length
chunk_length = 512

# Convert to Traditional Chinese or not
convert_to_traditional = False

# Extract text and split
pure_text = extract_and_save_text(
    srt_filename=output_subtitle_path,
    output_filename=f"./output-{suffix}.txt",
)

chunks = chunk_text(text=pure_text, max_length=chunk_length)

# Part4 - Summarization  Claude

def summarization(client, summarization_prompt, model_name="claude-3-sonnet-20240229", temperature=0.0, top_p=1.0, max_tokens=512):
    """
    (1) Objective:
        - Use the Claude API to summarize a given text.

    (2) Arguments:
        - client: Claude API client.
        - text: The text to be summarized.
        - summarization_prompt: The summarization prompt.
        - model_name: The model name, default is "claude-3-sonnet-20240229". You can refer to "https://docs.anthropic.com/claude/docs/models-overview#model-comparison" for more details.
        - temperature: Controls randomness in the response. Lower values make responses more deterministic, default is 0.0.
        - top_p: Controls diversity via nucleus sampling. Higher values lead to more diverse responses, default is 1.0.
        - max_tokens: The maximum number of tokens to generate in the completion, default is 512.

    (3) Return:
        - The summarized text.

    (4) Example:
        - If the text is "ABC" and the summarization prompt is "DEF", system_prompt is "GHI", model_name is "claude-3-sonnet-20240229",
          temperature is 0.0, top_p is 1.0, and max_tokens is 512, then you can call the function like this:

              summarization(client=client, text="ABC", summarization_prompt="DEF", system_prompt="GHI", model_name="claude-3-sonnet-20240229", temperature=0.0, top_p=1.0, max_tokens=512)

    """

    user_prompt = summarization_prompt

    while True:

        try:
            # Use the Claude API to summarize the text.
            message = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            break

        except:
            # If the API call fails, wait for 1 second and try again.
            print("The API call fails, wait for 1 second and try again.")
            time.sleep(1)

    return message.content[0].text

# claude api
# @title Parameter Setting of Claude { run: "auto" }
''' ===== In this block, you can modify your desired parameters and set your Claude API key ===== '''

# Your Claude API key.
# @markdown **claude_api_key**: Your Claude api key.
claude_api_key = "YOUR_CLAUDE_API_KEY" # @param {type:"string"}

# The model name, default is "claude-3-opus-20240229". You can refer to "https://docs.anthropic.com/claude/docs/models-overview#model-comparison" for more details.
# @markdown **model_name**: The model name, default is "claude-3-opus-20240229". You can refer to "https://docs.anthropic.com/claude/docs/models-overview#model-comparison" for more details.
model_name = "claude-3-opus-20240229" # @param {type:"string"}

# Controls randomness in the response. Lower values make responses more deterministic.
# @markdown **temperature**: Controls randomness in the response. Lower values make responses more deterministic.
temperature = 1 # @param {type:"slider", min:0, max:1, step:0.1}

# Controls diversity via nucleus sampling. Higher values lead to more diverse responses.
# @markdown **top_p**: Controls diversity via nucleus sampling. Higher values lead to more diverse responses.
top_p = 1.0 # @param {type:"slider", min:0, max:1, step:0.1}

# Prompt Setting of Claude Multi-Stage Summarization: Paragraph
# @title Prompt Setting of Claude Multi-Stage Summarization: Paragraph { run: "auto" }
''' You can modify the summarization prompt and maximum number of tokens. '''
''' However, DO NOT modify the part of <text>.'''

# The maximum number of tokens to generate in the completion.
# @markdown **max_tokens**: The maximum number of tokens to generate in the completion.
max_tokens = 350 # @param {type:"integer"}

# @markdown #### Changing **summarization_prompt_template**
# @markdown You can modify the summarization prompt and maximum number of tokens. However, **DO NOT** modify the part of `<text>`.
summarization_prompt_template = "Write a summary of this text within 300 characters, including key points and all important details：<text>" # @param {type:"string"}

#Step1: Split the long text into multiple smaller pieces and obtain summaries for each smaller text piece separately

paragraph_summarizations = []

# First, we summarize each section that has been split up separately.
for index, chunk in enumerate(chunks):

    # Record the start time.
    start = time.time()

    # Construct summarization prompt.
    summarization_prompt = summarization_prompt_template.replace("<text>", chunk)

    # We summarize each section that has been split up separately.
    response = summarization(client=client, summarization_prompt=summarization_prompt, model_name=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    # Calculate the execution time and round it to 2 decimal places.
    cost_time = round(time.time() - start, 2)

    # Print the summary and its length.
    print(f"----------------------------Summary of Segment {index + 1}----------------------------\n")
    for text in textwrap.wrap(response, 80):
        print(f"{text}\n")
    print(f"Length of summary for segment {index + 1}: {len(response)}")
    print(f"Time taken to generate summary for segment {index + 1}: {cost_time} sec.\n")

    # Record the result.
    paragraph_summarizations.append(response)

# First, we collect all the summarizations obtained before and print them.

collected_summarization = ""
for index, paragraph_summarization in enumerate(paragraph_summarizations):
    collected_summarization += f"Summary of segment {index + 1}: {paragraph_summarization}\n\n"

print(collected_summarization)

# 2 Prompt Setting of Gemini Multi-Stage Summarization: Total
# @title Prompt Setting of Gemini Multi-Stage Summarization: Total { run: "auto" }
''' You can modify the summarization prompt and maximum number of tokens. '''
''' However, DO NOT modify the part of <text>.'''

# We set the maximum number of tokens to ensure that the final summary does not exceed 550 tokens.
# @markdown **max_tokens**: We set the maximum number of tokens to ensure that the final summary does not exceed 550 tokens.
max_tokens = 550 # @param {type:"integer"}

# @markdown ### Changing **summarization_prompt_template**
# @markdown You can modify the summarization prompt and maximum number of tokens. However, **DO NOT** modify the part of `<text>`.
summarization_prompt_template = "Write a concise summary of the following text within 500 characters：<text>" # @param {type:"string"}

# Finally, we compile a final summary from the summaries of each section.

# Record the start time.
start = time.time()

summarization_prompt = summarization_prompt_template.replace("<text>", collected_summarization)

# Run final summarization.
final_summarization = summarization(client=client, summarization_prompt=summarization_prompt, model_name=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

# Calculate the execution time and round it to 2 decimal places.
cost_time = round(time.time() - start, 2)

# Print the summary and its length.
print(f"----------------------------Final Summary----------------------------\n")
for text in textwrap.wrap(final_summarization, 80):
    print(f"{text}")
print(f"\nLength of final summary: {len(final_summarization)}")
print(f"Time taken to generate the final summary: {cost_time} sec.")

''' In this block, you can modify your desired output path of final summary. '''

output_path = f"./final-summary-{suffix}-claude-multi-stage.txt"

# If you need to convert Simplified Chinese to Traditional Chinese, please set this option to True; otherwise, set it to False.
convert_to_tradition_chinese = False

if convert_to_tradition_chinese == True:
    # Creating an instance of OpenCC for Simplified to Traditional Chinese conversion.
    cc = OpenCC('s2t')
    final_summarization = cc.convert(final_summarization)

# Output your final summary
with open(output_path, "w") as fp:
    fp.write(final_summarization)

# Show the result.
print(f"Final summary has been saved to {output_path}")
print(f"\n===== Below is the final summary ({len(final_summarization)} words) =====\n")
for text in textwrap.wrap(final_summarization, 64):
    print(text)


# Prompt Setting of Claude Refinement 总结

# @title Prompt Setting of Claude Refinement { run: "auto" }
''' You can modify the summarization prompt and maximum number of tokens. '''
''' However, DO NOT modify the part of <text>.'''

# We set the maximum number of tokens.
# @markdown **max_tokens**: We set the maximum number of tokens.
max_tokens = 550 # @param {type:"integer"}

# @markdown ### Changing **summarization_prompt_template** and **summarization_prompt_refine_template**
# @markdown You can modify the summarization prompt and maximum number of tokens. However, **DO NOT** modify the part of `<text>`.

# Initial prompt.
# @markdown **summarization_prompt_template**: Initial prompt.
summarization_prompt_template = "Please provide a concise summary of the following text within 300 characters:<text>" # @param {type:"string"}

# Refinement prompt.
# @markdown **summarization_prompt_refinement_template**: Refinement prompt.
summarization_prompt_refinement_template = "Please provide a concise summary within 500 characters, combining the original summary and the new content:<text>" # @param {type:"string"}

paragraph_summarizations = []

# First, we summarize each section that has been split up separately.
for index, chunk in enumerate(chunks):

    if index == 0:
        # Record the start time.
        start = time.time()

        # Construct summarization prompt.
        summarization_prompt = summarization_prompt_template.replace("<text>", chunk)

        # Step1: It starts by running a prompt on a small portion of the data, generating initial output.
        first_paragraph_summarization = summarization(client=client, summarization_prompt=summarization_prompt, model_name=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        # Record the result.
        paragraph_summarizations.append(first_paragraph_summarization)

        # Calculate the execution time and round it to 2 decimal places.
        cost_time = round(time.time() - start, 2)

        # Print the summary and its length.
        print(f"----------------------------Summary of Segment {index + 1}----------------------------\n")
        for text in textwrap.wrap(response, 80):
            print(f"{text}\n")
        print(f"Length of summary for segment {index + 1}: {len(response)}")
        print(f"Time taken to generate summary for segment {index + 1}: {cost_time} sec.\n")


    else:
        # Record the start time.
        start = time.time()

        # Step2: For each following document, the previous output is fed in along with the new document.
        chunk_text = f"""Summary of the previous {index} paragraphs: {paragraph_summarizations[-1]}\nContent of paragraph {index + 1} : {chunk}"""

        # Construct refinement prompt for summarization.
        summarization_prompt = summarization_prompt_refinement_template.replace("<text>", chunk_text)

        # Step3: The LLM is instructed to refine the output based on the new document's information.
        paragraph_summarization = summarization(client=client, summarization_prompt=summarization_prompt, model_name=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        # Record the result.
        paragraph_summarizations.append(paragraph_summarization)

        # Calculate the execution time and round it to 2 decimal places.
        cost_time = round(time.time() - start, 2)

        # print results.
        print(f"----------------------------Summary of the First {index + 1} Segments----------------------------\n")
        for text in textwrap.wrap(paragraph_summarization, 80):
            print(f"{text}\n")
        print(f"Length of summary for the first {index + 1} segments: {len(paragraph_summarization)}")
        print(f"Time taken to generate summary for the first {index + 1} segments: {cost_time} sec.\n")

    # Step4: This process continues iteratively until all documents have been processed.

''' In this block, you can modify your desired output path of final summary. '''

output_path = f"./final-summary-{suffix}-claude-refinement.txt"

# If you need to convert Simplified Chinese to Traditional Chinese, please set this option to True; otherwise, set it to False.
convert_to_tradition_chinese = False

if convert_to_tradition_chinese == True:
    # Creating an instance of OpenCC for Simplified to Traditional Chinese conversion.
    cc = OpenCC('s2t')
    paragraph_summarizations[-1] = cc.convert(paragraph_summarizations[-1])

# Output your final summary
with open(output_path, "w") as fp:
    fp.write(paragraph_summarizations[-1])

# Show the result.
print(f"Final summary has been saved to {output_path}")
print(f"\n===== Below is the final summary ({len(paragraph_summarizations[-1])} words) =====\n")
for text in textwrap.wrap(paragraph_summarizations[-1], 80):
    print(text)
