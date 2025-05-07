import streamlit as st
import uuid
import asyncio 

from langchain_core.tracers.context import tracing_v2_enabled
from PyPDF2 import PdfReader 
import pandas as pd

from utils.generate import Generation
from utils.sidebar_config import setup_llm
from utils.llm_models import get_llm_model
from utils.embedding import VectorDB
import utils.utils as utils 
from config.llm_config import ChatConfig



def initialize_page():
    file_config = utils.load_config("src/config/file_config.yml")
    prompt_config = utils.load_config(file_config["llm_env"]["prompting_file"])
    model_config = utils.load_config(file_config["llm_env"]["model_config_file"])
    st.set_page_config(
        page_title="Text Generation",
    )
    st.title("Text Generation")
    return_para = {'model_config': model_config,
                   'prompt_config':prompt_config,
                   }
    st.markdown("""
    ##### *How to Use:*

    1. **Upload Your File** [[example file](https://docs.google.com/spreadsheets/d/11bME7vgxvwXotSUWK_itPfyQUqWfhTJD8TRzY58wcGI/edit?usp=drive_link)]
    """)
    return return_para

def handle_file_upload():
    file = st.file_uploader(
        "**Upload your file to create a document id**",
        type="xlsx",
        help="Excel file to be parsed",
    )
    return file

def main():
    return_para = initialize_page()
    model_config = return_para.get("model_config")
    prompt_config = return_para.get("prompt_config")
    
    # Start Generation bot 
    generate = Generation(prompt_config)
    # Start VectorDB
    vectordb = VectorDB(model_config)
    # Get all configuration settings
    config = setup_llm()
    # Overload config from streamlit UI
    model_config["condense_question"]["client"] = config['llm_choice']
    model_config["condense_question"]["model_name"] = config['model_choice']
    model_config["combine_docs"]["client"] = config['llm_choice']
    model_config["combine_docs"]["model_name"] = config['model_choice']
    # Access settings
    chat_permission = config['chat_permission']
    
    # initialize st session_state
    if 'process_btn' not in st.session_state:
        st.session_state.process_btn= None

    uploaded_file = handle_file_upload()   
    
    if uploaded_file is not None:
        process_btn = st.button("Read your excel",
                                    key="process_btn",
                                    help="Access Chatbot Read Your PDF to Gen Document ID")
        
        if process_btn:
            with st.spinner('Processing Excel file... This may take a moment'):
                try:
                    # Process excel file
                    # The 'uploaded_file' object itself can be passed to pd.read_excel
                    df = pd.read_excel(uploaded_file)

                    # --- THIS IS THE KEY PART TO SHOW THE DATAFRAME ---
                    st.subheader(f"Displaying data from: {uploaded_file.name}")
                    st.dataframe(df)  # Use st.dataframe for a nice interactive table
                    # Alternatively, you could use st.write(df) for a more static display

                    st.success(f'Excel file "{uploaded_file.name}" processed and displayed successfully!')

                except Exception as e:
                    st.error(f"Error processing Excel file: {e}")
                    st.error("Please ensure the uploaded file is a valid Excel format (.xlsx or .xls).")
    else:
        st.info("Once an Excel file is uploaded, click the 'Read your excel' button to view its content.")

    

    if not chat_permission:
        st.warning("Fill OpenAI API key") 
        return
    else:
        # Use a different key for session state
        if 'gen_chain_triggered' not in st.session_state:
            st.session_state.gen_chain_triggered = False

        gen_ex_quesiot_btn = st.button("Generate  Your Chain",
                                        help="Randomly generate an example question")

        if gen_ex_quesiot_btn:
            st.session_state.gen_chain_triggered = True

        if st.session_state.gen_chain_triggered:
            content = vectordb.choice_random_text(st.session_state.document_id2)
            llm = get_llm_model(
                chatmodel=config['llm_choice'], 
                model_name=config['model_choice'], 
                param=config['parameters'], 
            )
            
            question = generate.generate_question(llm, content)
            text = f'Here is the example question you can ask about the file: \n\n {question["text"]}'
            st.markdown(text)
            # Reset the trigger after displaying the question
            st.session_state.gen_chain_triggered = False
        
             
            
        
        if st.button("Re-generate"):
            st.session_state.messages2 = []
            st.session_state.session_id2 = str(uuid.uuid4())
            st.rerun()
            
        
        

if __name__ == "__main__":
    main()