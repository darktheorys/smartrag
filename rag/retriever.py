# Build a sample vectorDB
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_core.runnables import RunnableConfig

from llm import llm

# Load blog post
loader = UnstructuredHTMLLoader("data.html")
data = loader.load()

# Split
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
# splits = text_splitter.split_documents(data)

# VectorDB
# vectordb = Chroma.from_documents(documents=splits, embedding=embedder)


# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=vectordb.as_retriever(),
#     llm=llm.with_config(RunnableConfig(configurable={"model_kwargs": {"response_format": {"type": "text"}}})),
# )


# unique_docs = retriever_from_llm.get_relevant_documents(
#     query="Extract the numerical value associated with capital expenditures, ensuring it is represented in USD millions"
# )
print(data[0].page_content)

text = data[0].page_content

resp = llm.with_config(RunnableConfig(configurable={"model_kwargs": {"response_format": {"type": "text"}}})).invoke(
    f"{text}\nExtract the numerical value associated with capital expenditures, ensuring it is represented in USD millions."
)

print(resp)
