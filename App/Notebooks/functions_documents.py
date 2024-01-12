
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pandas as pd
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def dataframe_documentation(documents:list):
    data = pd.DataFrame()
    page_content = []
    metadata = []
    for line in range(len(documents)):
        page_content.append(documents[line].page_content)
        metadata.append(documents[line].metadata)
    data["page_content"] = page_content
    data["metadata"] = metadata
    return data

# Creando los nuevos embeddings 

def created_embeddings(data:object, embeddings=embedding_function):
    col_index = data.columns.get_loc('page_content')
    vectors = []
    for index in range(data.shape[0]):
        val = embeddings.embed_query(data.iloc[index, col_index])
        vectors.append(val)
    data["embeddings"] = vectors
    return data