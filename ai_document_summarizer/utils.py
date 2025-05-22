from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader
import os
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain



def process_text(text):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(separator= '\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text) #Splitting the text into chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #load a model for generating embeddings
    vectorstore = FAISS.from_texts(chunks, embeddings) #create faiss index
    return vectorstore

def summarizer(pdf):
    if pdf is not None:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        vectorstore = process_text(text)
        query = "Summarize the PDF document within 3-5 sentences."
        if query:
            docs = vectorstore.similarity_search(query)
            summarization_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_length=600)
            llm = HuggingFacePipeline(pipeline=summarization_pipeline)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as callback:
                response = chain.run(input_documents=docs, question=query)
                return response #return the summary