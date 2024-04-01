import json

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field
from tqdm import tqdm

from llm import llm
from models import QueryAmbiguation


class IntentExtraction(BaseModel):
    intent: str = Field(description="Intent of the query to be answered.")


output_parser = PydanticOutputParser(pydantic_object=IntentExtraction)

sys_message = """Extract the intent from given query as strings. It should help a person who is aiming to answer that question.
Queries may contain an ambiguous abbreviation, for them, abbreviation and possible disambiguations will be provided. Your task is not to select from them but to provide intent details.
Do not assume and output any full-form in the intent.

Domain of the query is {domain}.

{format_instructions}"""

user_message = """Query:{query}
Abbreviation:{abbrv}
Possible Disambiguations:{disambs}
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
            input_variables=["query", "abbrv", "disambs"],
            partial_variables={},
        )
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | output_parser


def extract_intent(df: pd.DataFrame, top_n: int, domain: str | None) -> None:
    for i in tqdm(range(len(df))):
        query = df.loc[i, "ambiguous_question"]
        ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))

        # focus on only the first ambiguity
        amb = ambiguities.full_form_abbrv_map[0]
        disambs = ""
        if not pd.isna(df.loc[i, f"top_{top_n}_full_form"]):
            full_forms: list[str] = json.loads(df.loc[i, f"top_{top_n}_full_form"])[0]
        else:
            full_forms: list[str] = json.loads(df.loc[i, "llm_full_form_suggestion"])
        disambs = "".join([f"{i} - {full_form}\n" for i, full_form in enumerate(full_forms)])

        answer: IntentExtraction = chain.invoke(
            {"query": query, "abbrv": amb.abbreviation, "disambs": disambs, "domain": domain if domain else ""}
        )

        df.loc[i, "intent"] = answer.intent
