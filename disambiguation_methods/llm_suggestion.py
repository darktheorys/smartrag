import json

import pandas as pd
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from tqdm import tqdm

from disambiguation_methods.domain_extractor import categories
from llm import llm
from models import QueryAmbiguation


class AbbrvResolution(BaseModel):
    full_form: str


output_parser = PydanticOutputParser(pydantic_object=AbbrvResolution)
output_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)


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


def get_abbreviation_suggestions(df: pd.DataFrame, top_n: int = 10, llm: str = "gpt4", temp: float = 1.0) -> None:
    with tqdm(range(len(df))) as pbar:
        for i in pbar:
            ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))
            ambiguous_question = df.loc[i, "ambiguous_question"]
            suggestions = []
            for amb in ambiguities.full_form_abbrv_map:
                answer1: AbbrvResolution = llm_suggestor.invoke(
                    {
                        "query": ambiguous_question,
                        "abbrv": amb.abbreviation,
                        "domain": categories[df.loc[i, "domain_idx"]][0]
                        if df.loc[i, "domain_idx"] < len(categories)
                        else None,
                    },
                    config=RunnableConfig(configurable={"llm": llm, "temperature": temp}),
                )
                suggestions.append(answer1.full_form)
                pbar.set_postfix_str(str(suggestions))
            df.loc[i, "llm_full_form_suggestions"] = json.dumps(suggestions)
