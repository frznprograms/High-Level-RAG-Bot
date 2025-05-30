import os
import yaml
from dotenv import load_dotenv
from retriever import Retriever
from data_loader import Dataloader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

cqsp = """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
        """

class Generator:
    def __init__(self, retriever, config_path=None):
        if config_path is None:
            # path to generator.py:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "../config.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        llm_config = config.get("llm", {})

        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", llm_config.get("azure_deployment")),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", llm_config.get("api_version")),
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens"),
            timeout=llm_config.get("timeout"),
            max_retries=llm_config.get("max_retries", 2),
            stream=llm_config.get("stream", False),
        )

        self.chat_history = []
        self.retriever = retriever.retriever
        self.rag_chain = None
        self.chat_history = []
        self.retriever = retriever.retriever
        if self.retriever is None: 
            raise ValueError("Retriever has not been initialised. Please pass an \
                             initialised Retriever to the Generator...")
        self.rag_chain = None
        
    def init_rag_chain(self, cqsp=cqsp):
        contextualize_q_system_prompt = cqsp
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to \
             answer the user's question."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        self.rag_chain = rag_chain

    def clear_history(self):
        reply = input("Are you sure you want to clear chat history? [y/n]")
        if reply.strip() == 'y':
            self.chat_history = []
        else: 
            print("Abandoning operation - chat history will be retained.")

    def query(self, query):
        answer = self.rag_chain.invoke({"input": query, "chat_history": self.chat_history})['answer']
        self.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer)
        ])
        return answer

if __name__ == "__main__":
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()

    retriever = Retriever()
    retriever.initialize(splits)
    generator = Generator(retriever=retriever)
    generator.init_rag_chain()

    ans1 = generator.query("When was Star Wars created?")
    print(ans1)
    ans2 = generator.query("What was its second franchise film?")
    print(ans2)
    generator.clear_history()
    print(generator.chat_history)
    