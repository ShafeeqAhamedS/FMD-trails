import os
import google.generativeai as genai # type: ignore
import logging
from dotenv import load_dotenv # type: ignore
from utils.helper import extract_code_cells, read_dataset_sample, read_file, prepare_environment, execute_python_file, get_prompt_execute_store
from utils.config import logger, generation_config

load_dotenv()

def main():
    try:
        genai.configure(api_key=os.getenv("API_KEY"))

        model = genai.GenerativeModel(
            model_name=os.getenv("MODEL_NAME"),
            generation_config=generation_config,
        )

        ipynb_file = "./User_Input/code.ipynb"
        dataset_file = "./User_Input/Position_Salaries.csv"

        user_input_path = "User_Input/*"
        backend_path = "../backend"

        prepare_environment(user_input_path, backend_path)

        eval_prompt_file = "./eval/prompt.md"
        eval_template_code = "./eval/template_code.md"
        eval_output_file = "./eval/output.py"
        eval_full_prompt_file = "./eval/full_prompt.md"
        eval_output_log = "./eval/output.log"
        replace_new_code_path = "./eval/new_code.py"

        backend_prompt_file = "./backend/prompt.md"
        backend_template_code = "./backend/template_code.md"
        backend_output_file = "./backend/output.py"
        backend_full_prompt_file = "./backend/full_prompt.md"  
        replace_backend_code_path = "/home/shafee/fmd/backend/main.py"

        frontend_prompt_file = "./frontend/prompt.md"
        frontend_template_code = "./frontend/template_code.md"
        frontend_output_file = "./frontend/output.jsx"
        frontend_full_prompt_file = "./frontend/full_prompt.md" 
        replace_frontend_code_path = "/home/shafee/fmd/frontend/src/App.jsx"
        
        logger.debug("Starting main execution")

        extracted_data = extract_code_cells(ipynb_file)
        df = read_dataset_sample(dataset_file)

        logger.info("Starting chat session")

        # Start a chat session 
        chat_session = model.start_chat() 

        get_prompt_execute_store(chat_session, eval_prompt_file, eval_template_code, extracted_data, df, eval_full_prompt_file, eval_output_file, replace_code_path=replace_new_code_path)

        # Run eval.py and get the output
        execute_python_file(eval_output_file, eval_output_log)

        eval_output = read_file(eval_output_log)

        get_prompt_execute_store(chat_session, backend_prompt_file, backend_template_code, extracted_data, df, backend_full_prompt_file, backend_output_file, eval_output, replace_backend_code_path)
        get_prompt_execute_store(chat_session, frontend_prompt_file, frontend_template_code, extracted_data, df, frontend_full_prompt_file, frontend_output_file, eval_output, replace_frontend_code_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()