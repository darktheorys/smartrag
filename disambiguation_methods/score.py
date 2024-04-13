from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field

from llm import llm


class AnswerJudge(BaseModel):
    bit1: int = Field(description="Bit representing first candidate", ge=0, le=1)
    bit2: int = Field(description="Bit representing second candidate", ge=0, le=1)
    bit3: int = Field(description="Bit representing third candidate", ge=0, le=1)


output_parser_score = PydanticOutputParser(pydantic_object=AnswerJudge)
output_parser_score = OutputFixingParser.from_llm(llm=llm, parser=output_parser_score)


sys_message_score = """Given a query, three candidate answers and one real answer, your task is to compare three candidate answers.

Each of the candidates are represented by the corresponding bit. Either 0 or 1. If candidate is well-enough and correct with respect to real answer it will be 1.
If candidate if not related with the answer or wrong, it will be 0.

Format instructions:
{format_instructions}
"""

user_message_score = """Query:{query}
Real Answer: {answer}
First Candidate: {amb}
Second Candidate: {unamb}
Third Candidate: {disamb}"""

messages_score = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message_score,
            input_variables=[],
            partial_variables={"format_instructions": output_parser_score.get_format_instructions()},
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=user_message_score,
            input_variables=["answer", "amb", "unamb", "disamb", "query"],
            partial_variables={},
        )
    ),
]
prompt_score = ChatPromptTemplate.from_messages(messages=messages_score)

chain_score = prompt_score | llm | output_parser_score
