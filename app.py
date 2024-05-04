import textwrap
from langchain_community.document_loaders import TextLoader
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

loader = TextLoader("data.txt")


document = loader.load()

# Preprocessing
def wrap_text_preserve_newlines(text, width=110):
    # Split input text into lines based on new lines
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline character
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


# Text splitting
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# print(docs)
# print(len(docs))

# Embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = "what are available loan schemes?"
doc = db.similarity_search(query)

# print(wrap_text_preserve_newlines(str(doc[0].page_content)))

# Q-A
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.8,"max_length":512})

chain=load_qa_chain(llm,chain_type="stuff")


queryText="what are available loan schemes?"


docResult=db.similarity_search(queryText)

chain.run(input_documents=docResult,question=queryText)