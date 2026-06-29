from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.5

)

chat = ChatHuggingFace(llm=llm, verbose=True)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def create_vector_db_from_youtube(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embedding_model)
    return db

def generate_pet_name(animal_type, pet_color):

    promt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} and I want a cool name for it, it is {pet_color} in color. "
        "Suggest five cool names for my pet."
    )

    pipeline = promt_template_name | llm
    response = pipeline.invoke({'animal_type': animal_type, 'pet_color': pet_color})
    return response

def langchain_agent():
    tools = load_tools(["wikipedia"], llm=llm)
    
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_kwargs={"handle_parsing_errors": True},
    )

    result = agent.invoke(
        "What is the avarage age of parrot? multiply the age by 3"
    )

    print(result)

if __name__ == "__main__":
    create_vector_db_from_youtube("https://www.youtube.com/watch?v=zvmVwFrV_IU")