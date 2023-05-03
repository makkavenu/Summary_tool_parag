import streamlit as st
#st.write("You are in Image generation tool page")

import time
import streamlit as st
import openai
openai.api_key = st.secrets["api_key"]
from st_pages import Page, show_pages, add_page_title

add_page_title("Image Generation Tool")

@st.cache(allow_output_mutation =True)
def request_image_model(description):
    try:
	    response = openai.Image.create(
	    	prompt = description,
	    	n = 1,
	    	size = "1024x1024"
	    	)

    except:
        time.sleep(10)
        return request_openai(question)

    return response


description = str(st.text_input('**Describe image you want to generate and hit Enter:** (NOTE: The more detailed the description, the more likely you are to get the result that you or your end user want.)'))
st.write("EXAMPLES: ")
st.write("1. a white siamese cat \n  2. an armchair in the shape of an avocado \n 3. A sunlit indoor lounge area with a pool containing a flamingo")
#st.write("NOTE: The more detailed the description, the more likely you are to get the result that you or your end user want.")

if description:
	response = request_image_model(description)
	answer = response["data"][0]["url"]
	st.markdown("RESPONSE: \n Click the below link to see the generated image ")
	st.write(answer)
