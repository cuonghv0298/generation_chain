from service.genbot import Simple_Assistance

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

class Generation:
    def __init__(
        self,
        prompt_config
    ):
        self.prompt_config = prompt_config
        self.default_param = {
            "temperature":0.7,
            "top_p":0.8, 
            "max_retries":2,
        }

    def generate_question(
        self,
        llm, 
        content, 
        prompt="",
        params = {},        
    ):
        # Set model para
        if params == {}:
            params = self.default_param
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_example_question_prompt']
        
        # Set llm para
        assistant_param = {
            "template": prompt,
            "content":  content,
            "llm": llm, 
        }

        bot = Simple_Assistance(**assistant_param)
        response = bot()
        return response
    def summarized_by_stuff(
            self,
            llm,
            docs
        ):
        
        prompt_template = self.prompt_config['prompt']["summarized_by_stuff_prompt"]
        prompt = PromptTemplate.from_template(prompt_template)
        stuff_chain = create_stuff_documents_chain(
                llm=llm, 
                prompt = prompt,
                document_variable_name = "text",
                )
        return stuff_chain.invoke({"text": docs})


    