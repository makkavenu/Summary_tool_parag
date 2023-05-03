import streamlit as st
import openai
from streamlit.components.v1 import html
from st_pages import Page, show_pages, add_page_title
#st.sidebar.image: st.sidebar.image("SkipText Offiical Logo.png", use_column_width=True)
add_page_title("Summarization Input Selection")

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

# if "email_id" in st.session_state:

# selection = str(st.selectbox("What do you want to do: ", ["None", "Summarize Text"]))
st.session_state["task_type"] = "Summarize Text"#selection
# def select_input_type():
#     input_type = str(st.selectbox("How will you provide text? ", ["None", "Enter Text", "Upload File"]))
#     return input_type

# if selection:
input_type = str(st.selectbox("How will you provide text? ", ["None", "Enter Text", "Upload File"]))
st.session_state["input_type"] = input_type
if input_type != "None":
    nav_page("Summarization%20Play%20Ground")
    # if input_type=="Enter Text":
    #     nav_page("two")
# elif selection=="Outline Text":
#     input_type = select_input_type()
# else:
#     st.write('''You are not logged in, Go to "Login page" and enter your email.''')
