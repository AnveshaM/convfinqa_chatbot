import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embeddings import get_embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    response_text, context_text = query_rag(query_text, None)

    # Display the response on the terminal
    formatted_response = f"Response: {response_text}"
    print(formatted_response)
    while True:
        # Ask if the user wants to continue querying
        choice = input("Do you want to ask some more questions from the same document? (yes/no): ").lower()
        if choice != "yes":
            return

        previous_context = context_text
        new_query_text = input("Please type your query: ")
        new_response_text, context_text = query_rag(new_query_text, previous_context)
        # Display the response on the terminal
        new_formatted_response = f"Response: {new_response_text}"
        print(new_formatted_response)


def query_rag(query_text: str, previous_context):
    if previous_context is None:
        # Prepare the DB.
        embedding_function = get_embeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=1)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
    else:
        context_text = previous_context
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)

    return response_text, context_text


if __name__ == "__main__":
    main()
