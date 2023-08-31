import os
from langchain.embeddings import OpenAIEmbeddings 
from langchain.document_loaders import PyPDFLoader 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter


os.environ["OPENAI_API_KEY"] = "your OpenAI API key"

# Load your PDF file
pdf_path = "path to your pdf file"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the pdf into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()

# Create a vector database to store embedddings of your file 
db = Chroma.from_documents(documents, embedding=embeddings)
db.persist()

# Using Buffer memory to keep the conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define and customize the template
template = """you can cutomize this prompt to your needs.
Chat History:
{chat_history}
Follow Up Input: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template=template
)


pdf_qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0.8),
    db.as_retriever(),
    memory=memory,
    # Uncomment below line if you want the model to return the document with the answer
    #return_source_documents=True,
    )

query = input("Please ask your question:")
result = pdf_qa({"question": query})
print("Answer:")
print(result["answer"])