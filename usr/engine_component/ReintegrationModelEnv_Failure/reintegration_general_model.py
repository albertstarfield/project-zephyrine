#!/usr/bin/python3

#Code to execute when idle to reintegrate or integrate UMA and Conversation to use it become the concept of "Long-term memory" for the Adelaide general_conversation model, we can replace it after that

#fetch from usr/storage/UMA_state.json as unsupervised learning (convert JSON to csv like ML standard)

#fetch from usr/storage/presistentInteractionMem.json as supervised Learning ( Convert JSON Human-like natural flow language into csv like LLM standard flipflop (user to assistant then user again loop) Format dictionary) for finetuning

#make sure the chunk/batch is small so it can be terminated more suddenly and have small batch so doesn't require a specialized Nvidia to do such thing

#---------------------------------------------------------------------------------------
## Dataset format to follow (https://huggingface.co/datasets/Open-Orca/OpenOrca) openorca


# Sample code on how to train LLM from scratch (Target pytorch)
# LLaMa-2 Finetune


### Debug Execution from Index.js Adelaide Engine!
# Find out the current Directory

# Find out the current virtual environment

# Find out the the listing of current directory

# Where is huggingface?

# Find out whether the path to the storage UMA_state is accessible and presistentInteractionMem.json


import os

# Set the TRANSFORMERS_CACHE environment variable
os.environ["TRANSFORMERS_CACHE"] = "./generalchat_modelkitchen"


# CUDA or CPU for now (I mean after all, the vram already being jacked into )
#----------------------------------------------------------------------
import pandas as pd
import json

# Load JSON data
with open('UMA.json', 'r') as file:
    data = json.load(file)

# Create an empty DataFrame
df = pd.DataFrame(columns=['text'])

# Iterate through the JSON data and append it to the DataFrame
for item in data:
    df = df.append({'text': item}, ignore_index=True)

# Display the DataFrame
print(df)







