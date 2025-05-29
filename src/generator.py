import os
from dotenv import load_dotenv
from retriever import Retriever
from data_loader import Dataloader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
    def __init__(self, retriever):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.chat_history = []
        self.retriever = retriever.retriever
        if self.retriever is None: 
            raise ValueError("Retriever has not been initialised. Please pass an initialised Retriever to the Generator...")
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
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        self.rag_chain = rag_chain

if __name__ == "__main__":
    loader = Dataloader()
    documents = loader.load_documents()
    splits = loader.chunk_documents()

    retriever = Retriever()
    retriever.initialize(splits)
    generator = Generator(retriever=retriever)
    generator.init_rag_chain()
    
    question1 = "When was Star Wars created?"
    answer1 = generator.rag_chain.invoke({"input": question1, "chat_history": generator.chat_history})['answer']
    generator.chat_history.extend([
        HumanMessage(content=question1),
        AIMessage(content=answer1)
    ])

    print(f"Human: {question1}")
    print(f"AI: {answer1}\n")

    question2 = "What was its second franchise film?"
    answer2 = generator.rag_chain.invoke({"input": question2, "chat_history": generator.chat_history})['answer']
    generator.chat_history.extend([
        HumanMessage(content=question2),
        AIMessage(content=answer2)
    ])

    print(f"Human: {question2}")
    print(f"AI: {answer2}")