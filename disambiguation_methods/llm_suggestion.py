import json

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from tqdm import tqdm

from llm import llm
from models import QueryAmbiguation


class AbbrvResolution(BaseModel):
    full_form: str


output_parser = PydanticOutputParser(pydantic_object=AbbrvResolution)


sys_message = """Find the full form of the asked abbreviation in the respective query.
Domain of the questions is {domain}.

{format_instructions}"""

user_message = """Abbreviation: {abbrv}
Query: {query}
Output:"""


messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message,
            input_variables=["domain"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=user_message,
            input_variables=["query", "abbrv"],
            partial_variables={},
        )
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)


llm_suggestor = prompt | llm | output_parser


def get_abbreviation_suggestions(df: pd.DataFrame, top_n: int = 10) -> None:
    for i in tqdm(range(len(df))):
        ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))
        ambiguous_question = df.loc[i, "ambiguous_question"]
        suggestions = []
        for amb in ambiguities.full_form_abbrv_map:
            answer1: AbbrvResolution = llm_suggestor.invoke(
                {"query": ambiguous_question, "abbrv": amb.abbreviation, "domain": df.loc[i, "domain"]}
            )
            suggestions.append(answer1.full_form)
        df.loc[i, "llm_full_form_suggestions"] = json.dumps(suggestions)
