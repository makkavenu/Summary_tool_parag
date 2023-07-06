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
from pydub import AudioSegment


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
            # audio_file = open(audio_path, "rb")
            song = AudioSegment.from_mp3(audio_path)
            # st.write("len(song): ", len(song))
            five_minutes = 1 * 60 * 1000 #1 minute
            if len(song)<five_minutes:
                audio_file = open(audio_path, "rb")
                result = openai.Audio.translate("whisper-1", audio_file)
                text = result["text"]
            else:
                overlap = 1 * 1000 #3 sec
                transcript_chunks = []
                for i in range(five_minutes, len(song), five_minutes):
                    #st.write("i: ", i)
                    if i==five_minutes:
                        start = 0
                    else:
                        start = i-five_minutes - overlap
                    current_5_mins = song[start:i]
                    #st.write("start, i: ", start, i)
                    #st.write("len: ", len(current_5_mins))
                    audio_path_chunk = f"audio_{i}.wav"
                    current_5_mins.export(audio_path_chunk, format="wav")
                    audio_file_chunk = open(audio_path_chunk, "rb")

                    current_result = openai.Audio.translate("whisper-1", audio_file_chunk)
                    transcript_chunks.append(current_result["text"])
                    #st.write("current_result***&&&^^^:", current_result)
                    os.remove(audio_path_chunk)

                text = ", ".join(transcript_chunks)
            st.success("Audio converted to Text!")
            #st.write("Text: ", result["text"])
            # text = result["text"]
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
