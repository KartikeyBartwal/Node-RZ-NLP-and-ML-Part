import os
from langchain_community.llms import Replicate
from langchain.prompts import PromptTemplate 
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


class LoadModel:
    model = None 
    def load_local_model():
        pass 

    def use_hosting():
        model_name = "meta/codellama-13b:511fc67df70ee2d584375b6f1463d8d7d9ca7e6131e0f0a879d32d99bce17351"
        replicate_api_token = "r8_1tXV1Jxf2JBIiunu0PrGCEuYgOabv251OjQ0x"
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
        llm = Replicate(
            model= model_name,
            input={
                "top_k": 250,
                "top_p": 0.95,
                "max_tokens": 500,
                "temperature": 0.95,
                "repeat_penalty": 1.1,
                "presence_penalty": 0,
                "frequency_penalty": 0
            },
            auth_token=replicate_api_token  
        )

        return llm
        
    def use_model(llm , prompt):
        output = llm(prompt)

        formatted_output = ""
        for i in range(1, len(output), 1):
            formatted_output += output[i] 
        
        return formatted_output
        