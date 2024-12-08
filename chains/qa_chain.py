from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from .context_retriever import ContextRetriever

class QAChain:
    
    def __init__(self, context_retriever: ContextRetriever):
        
        self.retriever = context_retriever.retriever
        self.chain = self._create_qa_chain()
    
    def query(self, user_message: str) -> str:
        """
        Processes a user's message and generates an appropriate response.

        :param user_message: The message from the user
        :return: A ChatBotResponse object containing the system's response
        """

        try:
            response = self.chain.invoke({
                "input": user_message
            })
            response = response["answer"]

            return response
            
        except Exception as e:
            print(f"Failed to query the QA chain: {str(e)}")

    def _create_qa_chain(self):
        """
        Creates a conversation chain for processing user input and generating responses.

        :return: A LangChain retrieval chain
        """
        try:
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

            system_prompt = (
                "You are a helpful assistant answering questions about the provided code. "
                "Use the following pieces of retrieved context to answer the question. "
                "Do not provide code in your responses unless given explicit instructions to do so. \n\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("user", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt)

            return create_retrieval_chain(self.retriever, question_answer_chain)

        except Exception as e:
            print(f"Failed to create conversation chain {str(e)}")