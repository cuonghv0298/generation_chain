from langchain.prompts import PromptTemplate
from langchain import callbacks

class Simple_Assistance:
    """
    From the defined template and input content to generate text
    Arg:
    - template: prompt confige
    - content: text input
    """

    def __init__(
        self,
        template,
        content,
        llm,
    ):
        prompt = PromptTemplate.from_template(
            template
        )
        prompt.format(
            content=content,
        )
        self.llm_chain = prompt | llm
        self.content =  content
        
    def __call__(self):        
        with callbacks.collect_runs() as cb:
            result = self.llm_chain.invoke(
                {
                    "content": self.content,
                },
            )
            run_id= cb.traced_runs[0].id
        response = {
                "text": result.content,
                "__run":{
                    "run_id": run_id
                }
            }
        return response
    