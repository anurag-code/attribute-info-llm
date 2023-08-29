import os
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import requests
import openai
import json
import io
import glob
import tqdm
import chromadb
import tiktoken

# establish conections
with open('C:/Users/anuni/OneDrive/Desktop/config/config_key.json','r') as key:
    dict_key=json.load(key)

openai.api_key=dict_key['OPENAI_API_KEY']

chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "What is the capital of India"}])
# check
print(chat_completion['choices'][0]['message']['content'])

# check using langchain libraries
llm = OpenAI(temperature=0,openai_api_key=dict_key['OPENAI_API_KEY'])

print(llm.predict("what is the capital of Japan"))

# Create doc
path_dbt='C:\coding\llm-test\jaffle_shop'

# Read the SQL file
def read_sql_file(path):
    with open(path, "r") as sql_file:
        sql_content = sql_file.read()
    return sql_content


# Convert SQL to English sentences 
def sql_to_english(sql_code):
    
    prompt = f"""Explain this SQL code in such a way that it can be understood by a business person, who doesnt have knowledge of SQL. 
    Also, all the attributes names and joins and transformations should be explained. If there are any Sub Queries or CTEs. 
    Then please explain each of them seperately and explain how they are used subsequently in final table. 
    If there is some transformation on an attribute 'A' and that has beed aliased as 'B', then include both the names as it is in the explanation.
    Please keep the names of attributes same as given in SQL file , for the explanation.
    :\n{sql_code}"""
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can also use other engines
        prompt=prompt,
        max_tokens=500,  # Adjust to control the response length
        temperature=0
    )
    
    english_text = response.choices[0].text.strip()
    return english_text


# initialize dict
doc_dict={}

# cretae dictionary for each sql file
for root,_,files in os.walk(path_dbt,topdown=False):
    for name in files:
        if '.sql' in name.lower():
            filepath=os.path.join(root,name)
            sql_content = read_sql_file(filepath)
            english_text = sql_to_english(sql_content)
            # append dict
            doc_dict[name]=english_text 


# convert dict to json
json_object_string = json.dumps(doc_dict)

# Write data to text file
with open('C:/coding/text-file/api_key.txt', "w") as text_fl:
    text_fl.write(json_object_string)


# Document Loading
from langchain.document_loaders import TextLoader

loader = DirectoryLoader("C:/coding/text-file", glob="**/*.txt", loader_cls=TextLoader)

data = loader.load()

print('\n loaded data : \n',data)

# Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a  small chunk size, just to chk.
    chunk_size = 250,
    chunk_overlap  = 50,
    length_function = len,
#     separators=["\n\n", "\n", "\. ", " ", ""]
)

docs = text_splitter.split_documents(data)

print('\n splitted text: \n',docs )


# Vectordb and Embedding

# Embeddings
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Vector Store
persist_directory ="C:\coding\chroma_db"

vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)

print("successfully created vectordb")

# Retrieval
llm_r = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=dict_key['OPENAI_API_KEY'])

qa_chain = RetrievalQA.from_chain_type(llm_r,retriever=vectordb.as_retriever())
result=qa_chain({"query": 'How the attribute first_order has been calculated? Please explain its derivation in detail along with the details about the dependeant attributes from the begining.'})
print('\n\n Result : \n',result["result"])