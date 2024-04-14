from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel

from llm import llm


class AnswerStr(BaseModel):
    answer: str


class AnswerBool(BaseModel):
    answer: bool


IS_BOOL = False

output_parser_bool = PydanticOutputParser(pydantic_object=AnswerBool)
output_parser_bool = OutputFixingParser.from_llm(llm=llm, parser=output_parser_bool)

output_parser_str = PydanticOutputParser(pydantic_object=AnswerStr)
output_parser_str = OutputFixingParser.from_llm(llm=llm, parser=output_parser_str)

output_parser_answer = output_parser_str if not IS_BOOL else output_parser_bool

sys_message_answer = """Answer the given questions as concise and short as possible. Do not output something else.
Additionally, an intent and requirements for the answer can be provided by the user. Take them into consideration while answering.
Domain of the questions is {domain}.

{format_instructions}
"""

user_message_answer = """Query:{query}
Intent:{intent}
Expected Dtype:{dtype}
Output:"""


messages_answer = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message_answer,
            input_variables=["domain"],
            partial_variables={"format_instructions": output_parser_answer.get_format_instructions()},
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=user_message_answer, input_variables=["query", "intent", "dtype"], partial_variables={}
        )
    ),
]
prompt_answer = ChatPromptTemplate.from_messages(messages=messages_answer)

chain_answer = prompt_answer | llm | output_parser_answer
