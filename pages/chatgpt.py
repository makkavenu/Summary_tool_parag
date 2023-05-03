import streamlit as st
import openai
import os
openai.api_key = st.secrets["api_key"]
os.environ['OPENAI_API_KEY']=openai.api_key
from st_pages import Page, show_pages, add_page_title

add_page_title("ChatGPT Tool")

@st.cache(allow_output_mutation =True)
def asking_chatGPT_model(system_content, user_question_content):
    # st.write("inside ask_gpt_3_turbo_model new...")
    # system_content = '''You are a chatbot that can answer all User's questions about Pharma."'''
    # user_question_content = '''Generate a Conversational Answer to User Question as truthfully as possible using the provided Context. If you don't find exact answer in Context say "I don't have enough data to Answer this Question. \nContext:\n{} \nUser Question: {} \n'''.format(context, question)
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            #{"role": "system", "content": system_content},
            {"role": "user", "content": str(user_question_content)},],
        temperature = 0,
        max_tokens = 300,
    )
    output_response = str(response["choices"][0]["message"]["content"])
    output_response = output_response.replace('$', '\$')
    return output_response


description = str(st.text_area('*Give your Prompt here*:',value ='''''', height = 300))#str(st.text_input('**Give your Prompt here and Hit Enter:**'))
with st.form("form6"):
	submit = st.form_submit_button(label = 'Submit')
if submit:
    output_response = asking_chatGPT_model("", description)
    st.write("**OUTPUT**:")
    st.write(output_response)
