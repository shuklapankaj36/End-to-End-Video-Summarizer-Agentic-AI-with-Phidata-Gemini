import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time 
from pathlib import Path
import tempfile
from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


#Page configuration

st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="https://th.bing.com/th/id/OIP.JLa6PUrs3cpXmoCViUwDaAHaHa?w=210&h=210&c=7&r=0&o=5&dpr=1.3&pid=1.7",
    layout="wide"
)

st.title("Phidata Video AI Summarizer agent")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initalize_agent():
    return Agent(
        name = "Video AI summarizer",
        model=Gemini(id = "gemini-2.0-flash-exp"),
        tools = [DuckDuckGo()],
        markdown = True,
    )

## Initialize the agent
multimodal_Agent=initalize_agent()

#file uploader
video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help = "Upload a file upto 200mb for AI analysis"

)

if video_file:
    with tempfile.NamedTemporaryFile(delete= False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    

    user_query = st.text_area(
        label="Enter your query:",
        placeholder="Type your query here...",
        help="Provide a multi-line query or input that will be processed.",
        height=200  # Optional: Specifies the height of the text area
    )

    if st.button(" Analyze Video", key="analyze_video_buttom"):
        if not user_query:
            st.warning("Please enter a query before analyzing the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights...."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = ( 
                        f"""You are a Video analyser agent. 
                        Analyze the uploaded video for content and context. 
                        Respond to the following query using video insights and supplementary web research.

                        {user_query}
                        
                        provide a detailed, user-friendly and actionable response. 
                            
                        """
                    )
                    #Ai agent processing
                    response = multimodal_Agent.run(analysis_prompt, video=[processed_video])

                #Display the result
                st.subheader("Analysis Result")
                st.success("Video processed successfully!")
                st.text_area("Analysis Results", analysis_prompt, height=200)

            except Exception as e:
                st.error(f"An error occurred during video analysis: {e}")
            finally:
                # clean a temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

# Customize text area height

st.markdown(
    """ 
    <style>
    .stTextArea textarea {
        height: 100px;

    }
    </style>

    """,
    unsafe_allow_html=True
)