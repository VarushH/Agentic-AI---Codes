import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.messages import HumanMessage,SystemMessage
from IPython.display import Image,display
from langgraph.graph import StateGraph,START,END
# from pydantic import BaseModel
from typing_extensions import TypedDict

st.title("Blog :blue[Generation]")

# st.write("Kindly drop your input here")

user_input = st.text_input("Kindly drop your input here")

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model ='meta-llama/llama-4-scout-17b-16e-instruct')


class OverallState(TypedDict):
    question:str
    title:str
    content:str

llm_refine = llm.with_structured_output(OverallState)

def create_title_node(state:OverallState):
    print('Starting title ceration')
    question = state['question']
    title = llm.invoke([
                       SystemMessage(content = 'You are a title generator for a Blog post. Return only one short Title for the blog based on user query: '),
                        HumanMessage(content = question),
                        ])

    return {'title': title.content}

def create_content_node(state:OverallState):
    print('Starting Content Creation')
    title = state['title']

    # content = llm.invoke(f'You have a generate a blog  for this title: {title}')
    content = llm.invoke([
                       SystemMessage(content = 'You are a content generator for a Blog post based on the title. Return only the content for blog based on title provided: '),
                        HumanMessage(content = title),
                        ]
    )
    return {'content':content.content}

#build workflow

workflow = StateGraph(OverallState)

#Add nodes
workflow.add_node('generate_title',create_title_node)
workflow.add_node('create_content',create_content_node)

#Add edges
workflow.add_edge(START,'generate_title')
workflow.add_edge('generate_title','create_content')
workflow.add_edge('create_content',END)


#Compile Graph
graph_flow = workflow.compile()


#Show workflow
# display(Image(graph.get_graph().draw_mermaid_png()))
if user_input:
    resp = graph_flow.invoke({'question': user_input })
    
    # st.markdown("<h1 style='text-align: center; color: grey;'>" + resp['title'] + "</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>" + resp['title'] + "</h2>", unsafe_allow_html=True)

    st.write(resp['content'])