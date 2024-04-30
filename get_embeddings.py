from langchain_community.embeddings import GPT4AllEmbeddings

def get_embeddings():

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = GPT4AllEmbeddings()

    return embeddings



