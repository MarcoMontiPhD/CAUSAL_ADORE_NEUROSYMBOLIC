import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


OLLAMA_MODEL = "deepseek-r1:8b"

SCIENTIFIC_FIELDS = ["biology", "chemistry", "physics", "computer science"]# to be expanded w/ more fields, possibly LLM generated as well

OUTPUT_DIR = "ontologies"


def initialize_ollama_llm():
    """Initializes and returns the Ollama LLM."""
    try:
        ollama_llm = Ollama(model=OLLAMA_MODEL)
        print(f"Successfully initialized Ollama LLM with model: {OLLAMA_MODEL}")
        return ollama_llm

    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        return None


def create_ontology_chain(llm):
    """Creates a LangChain chain to generate an ontology for a given field."""
    prompt_template = """
    You are an expert in creating ontologies for scientific fields.
    Generate a concise ontology for the field of {field} in RDF Turtle format.
    The ontology should define the main classes, properties, and relationships.
    Provide only the ontology in your response.

    Ontology for {field}:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["field"])
    return LLMChain(llm=llm, prompt=prompt)


def retrieve_and_store_ontology(chain, field, llm_name):
    """Retrieves an ontology for a given field and stores it in a file."""
    try:
        print(f"Querying {llm_name} for the '{field}' ontology...")
        ontology = chain.invoke({"field": field})["text"]

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        filename = os.path.join(OUTPUT_DIR, f"{field}_{llm_name}_ontology.ttl")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(ontology)

        print(f"Successfully stored the '{field}' ontology from {llm_name} in '{filename}'")

    except Exception as e:
        print(f"Failed to retrieve or store the ontology for '{field}' from {llm_name}: {e}")


def main():
    """Main function to orchestrate the ontology retrieval process."""
    ollama_llm = initialize_ollama_llm()

    if ollama_llm:
        ollama_chain = create_ontology_chain(ollama_llm)

        for field in SCIENTIFIC_FIELDS:
            retrieve_and_store_ontology(ollama_chain, field, "ollama")


if __name__ == "__main__":
    main()