import streamlit as st
from moviepy.editor import VideoFileClip
import os
import whisper

# from llama_index import SimpleDirectoryReader,LLMPredictor, PromptHelper
# from gpt_index import SimpleDirectoryReader,LLMPredictor, PromptHelper
from langchain import OpenAI
import openai
import time
import json
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import os


@st.cache_resource
def get_model(label):
    #st.write("Loading model......")
    model = whisper.load_model("small")
    return model

@st.cache_data
def recurisive_summarization(text):
    # text = uploaded[list(uploaded.keys())[0]]
    # text = str(text)
    text_splitter = TokenTextSplitter(chunk_size=2500, chunk_overlap=20)

    text_data = text
    texts= text_splitter.split_text(text_data)
    docs = [Document(page_content=t) for t in texts]
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY IN ENGLISH:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in English."

    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(OpenAI(temperature=0.7, max_tokens = 800), chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt)
    s = chain({"input_documents": docs}, return_only_outputs=True)
    summary_text= s['output_text']
    return summary_text

# @st.cache_data
def convert_video_to_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "44100", "-ac", "1"])
    # video.audio.write_audiofile(audio_path)

def main():
    st.title("Video Summarization Tool")

    # File upload section
    st.header("Upload a video file in mp4 or mov formats:")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video file temporarily
        video_path = uploaded_file.name #uploaded_file#"file_example_MP4_480_1_5MG.mp4" #"temp_video.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(uploaded_file.read())

        # Convert the video to audio
        audio_path = "output_audio.wav"
        with st.spinner("Converting Video to Audio....."):
            convert_video_to_audio(video_path, audio_path)

        # Provide download link for the audio file
        st.success("Video converted to audio!")
        st.audio(audio_path, format="audio/wav")

        #model = get_model(1)
        with st.spinner("Converting Audio to Text....."):
            #result = model.transcribe(audio_path)
            audio_file = open(audio_path, "rb")
            result = openai.Audio.translate("whisper-1", audio_file)
            st.success("Audio converted to Text!")
            #st.write("Text: ", result["text"])
            text = result["text"]
        with st.spinner("Generating Summary......"):
            summary = recurisive_summarization(text)
        st.success("Summary created!")
        st.subheader("**Output Summary:**")
        st.write(summary)
        # Delete the temporary video file
        os.remove(video_path)
        os.remove(audio_path)

if __name__ == "__main__":
    main()






# import streamlit as st

# from sklearn.datasets import make_moons
# from sklearn.tree import DecisionTreeClassifier
# # from dtreeviz.trees import dtreeviz
# # import dtreeviz.dtreeviz as dtreeviz
# import streamlit as st
# import graphviz as graphviz
# import matplotlib.pyplot as plt
# import streamlit.components.v1 as components

# st.set_option('deprecation.showPyplotGlobalUse', False)

# X, y = make_moons(n_samples=20, noise=0.25, random_state=3)

# treeclf = DecisionTreeClassifier(random_state=0)
# treeclf.fit(X, y)

# viz= dtreeviz(treeclf, X, y, target_name="Classes",

#     feature_names=["f0", "f1"], class_names=["c0", "c1"])

# def st_dtree(plot, height=None):

#     dtree_html = f"<body>{viz.svg()}</body>"

#     components.html(dtree_html, height=height)

# st_dtree(dtreeviz(treeclf, X, y, target_name="Classes",feature_names=["f0", "f1"], class_names=["c0", "c1"]),800)


# class Node:
#     def __init__(self, name, description, children=None):
#         self.name = name
#         self.description = description
#         self.children = children or []


# def render_node(node):
#     st.write(f"Node: {node.name}")
#     st.write(f"Description: {node.description}")

#     if node.children:
#         for child in node.children:
#             if st.checkbox(f"Expand {child.name}", key=child.name):
#                 render_node(child)


# def main():
#     root = Node("Root Node", "This is the root node.")
#     child1 = Node("Child 1", "This is child node 1.")
#     child2 = Node("Child 2", "This is child node 2.")
#     grandchild1 = Node("Grandchild 1", "This is grandchild node 1.")
#     grandchild2 = Node("Grandchild 2", "This is grandchild node 2.")

#     child1.children = [grandchild1]
#     child2.children = [grandchild2]
#     root.children = [child1, child2]

#     st.title("Interactive Tree Visualization")
#     st.write("Click on a node to expand its children.")
#     render_node(root)


# if __name__ == "__main__":
#     main()




# import streamlit as st


# class Node:
#     def __init__(self, name, description, children=None):
#         self.name = name
#         self.description = description
#         self.children = children or []


# def render_node(node):
#     st.write(f"Node: {node.name}")
#     st.write(f"Description: {node.description}")

#     if node.children:
#         for child in node.children:
#             if st.button(f"Expand {child.name}"):
#                 render_node(child)


# def main():
#     root = Node("Root Node", "This is the root node.")
#     child1 = Node("Child 1", "This is child node 1.")
#     child2 = Node("Child 2", "This is child node 2.")
#     grandchild1 = Node("Grandchild 1", "This is grandchild node 1.")
#     grandchild2 = Node("Grandchild 2", "This is grandchild node 2.")

#     child1.children = [grandchild1]
#     child2.children = [grandchild2]
#     root.children = [child1, child2]

#     st.title("Interactive Tree Visualization")
#     st.write("Click on a node to expand its children.")
#     render_node(root)


# if __name__ == "__main__":
#     main()







# import streamlit as st
# from graphviz import Digraph


# class Node:
#     def __init__(self, name, description, children=None):
#         self.name = name
#         self.description = description
#         self.children = children or []

#     def add_child(self, node):
#         self.children.append(node)


# def render_tree(node):
#     graph = Digraph()
#     render_node(graph, node)
#     st.graphviz_chart(graph.source)


# def render_node(graph, node):
#     graph.node(node.name, label=node.name, shape='box', style='filled', fillcolor='lightblue',
#                tooltip=node.description)
#     for child in node.children:
#         graph.edge(node.name, child.name)
#         render_node(graph, child)


# def main():
#     root = Node("Root Node", "This is the root node summary")
#     child1 = Node("Child 1", "This is child node 1 summary.")
#     child2 = Node("Child 2", "This is child node 2 summary")
#     grandchild1 = Node("Grandchild 1", "This is grandchild node 1 ")
#     grandchild2 = Node("Grandchild 2", "This is grandchild node 2.")

#     child1.add_child(grandchild1)
#     child2.add_child(grandchild2)
#     root.add_child(child1)
#     root.add_child(child2)

#     st.title("Interactive Tree Visualization")
#     st.write("Hover over a node to see summary.")
#     render_tree(root)


# if __name__ == "__main__":
#     main()










# # import streamlit as st
# # import json
# # import altair as alt

# # tree_data = {
# #     "name": "Root",
# #     "children": [
# #         {
# #             "name": "Node 1",
# #             "children": [
# #                 {"name": "Leaf 1"},
# #                 {"name": "Leaf 2"}
# #             ]
# #         },
# #         {
# #             "name": "Node 2",
# #             "children": [
# #                 {"name": "Leaf 3"},
# #                 {"name": "Leaf 4"}
# #             ]
# #         }
# #     ]
# # }


# # # def display_tree(tree_data):
# # #     # Convert tree data to JSON format
# # #     json_data = json.dumps(tree_data)

# # #     # Create D3.js visualization
# # #     chart = alt.Chart(data=alt.Data(values=json_data)).mark_circle().encode(
# # #         x='x:Q',
# # #         y='y:Q',
# # #         size='size:Q',
# # #         color='color:N',
# # #         tooltip='name:N'
# # #     ).transform_calculate(
# # #         x='datum.x * 400',
# # #         y='datum.depth * 100',
# # #         size='datum.value * 100',
# # #         color='datum.depth'
# # #     )

# # #     # Embed the chart in Streamlit
# # #     # st.write(chart)
# # #     st.altair_chart(chart, use_container_width=True)

# # # def display_tree(tree_data):
# # #     # Convert tree data to JSON format
# # #     json_data = json.dumps(tree_data)

# # #     # Create Vega-Lite specification for the tree visualization
# # #     chart = alt.Chart(alt.Data(values=json_data)).transform_calculate(
# # #         x='datum.x * 400',
# # #         y='datum.depth * 100',
# # #         size='datum.value * 100',
# # #         color='datum.depth'
# # #     ).mark_circle().encode(
# # #         x='x:Q',
# # #         y='y:Q',
# # #         size='size:Q',
# # #         color='color:N',
# # #         tooltip='name:N'
# # #     )

# # #     # Embed the chart in Streamlit
# # #     st.altair_chart(chart, use_container_width=True)

# # def display_tree(tree_data):
# #     # Convert tree data to pandas DataFrame
# #     df = pd.json_normalize(tree_data, "children", ["name"])

# #     # Create Vega-Lite specification for the tree visualization
# #     chart = alt.Chart(df).transform_calculate(
# #         x='datum.x * 400',
# #         y='datum.depth * 100',
# #         size='datum.value * 100',
# #         color='datum.depth'
# #     ).mark_circle().encode(
# #         x='x:Q',
# #         y='y:Q',
# #         size='size:Q',
# #         color='color:N',
# #         tooltip='name:N'
# #     )

# #     # Embed the chart in Streamlit
# #     st.altair_chart(chart, use_container_width=True)



# # display_tree(tree_data)


# # # if "satisfied" not in st.session_state:
# # #     st.session_state.satisfied = ''

# # # if st.session_state.satisfied:
# # #     st.write(" satisfied")

# # # left,middle,right,unused = st.columns([2,1,1,5], gap = "small")
# # # with left:
# # #     st.write("Answer Quality: ")
# # # with middle:
# # #     satisfied = st.button("👍")
# # #     st.write(satisfied)
# # #     st.session_state.satisfied = satisfied
# # # with right:
# # #     unsatisfied = st.button("👎")
