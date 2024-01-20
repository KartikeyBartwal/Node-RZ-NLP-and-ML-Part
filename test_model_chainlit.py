import chainlit as cl
import os
os.environ["REPLICATE_API_TOKEN"] = "r8_1tXV1Jxf2JBIiunu0PrGCEuYgOabv251OjQ0x"
print(os.environ.get("REPLICATE_API_TOKEN"))
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


llm = Replicate(
    model="meta/codellama-13b:511fc67df70ee2d584375b6f1463d8d7d9ca7e6131e0f0a879d32d99bce17351",
    input={"temperature": 0.75,
           "max_length": 500,
           "top_p": 1},
)

def use_my_model():
    prompt = input()
    output = llm(prompt)
    
    for i in range(1, len(output), 1):
        print(output[i] , end = "")    
        
    return output

@cl.on_chat_start
async def on_chat_start():
    #load model
    llm = Replicate(
    model="meta/codellama-13b:511fc67df70ee2d584375b6f1463d8d7d9ca7e6131e0f0a879d32d99bce17351",
    input={"temperature": 0.75,
           "max_length": 500,
           "top_p": 1},
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a chat json. The only thing you do is output me flow.json for my node-red application. You are well versed in node-red and no matter what I say, ask or talk about, the answer is always going to be in the json format. Nothing, just json format.'''),
            ("human", "{question}"),
        ]
    )
    #the initial prompt

    runnable = prompt | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content = "")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    # Send a response back to the user
        await msg.send()


