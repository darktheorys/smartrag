import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from tqdm import tqdm

from llm import llm
from models import QueryAmbiguation

sys_message = """Given a query from a question-answer dataset, your task is to identify abbreviations or possible abbreviations contained in the query.

- Extract everything as is, without changing a single thing, case, punctuation, encoding, anything.
- If there is an explicit abbreviation in the query, type is "abbreviation" and other fields are abbreviation itself and its full-form.
- If there is an implicit abbreviation (full-form) in the query, type is "full_form" and other fields are full-form itself and its abbreviation.

Example:
Query: Did Jack Dempsey fight the current world boxng council heavyweight champion in the US?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "full_form", "abbreviation": "WBC", "full_form": "world boxng council"}}, {{"ambiguity_type": "abbreviation", "abbreviation": "US", "full_form": "United States"}}]}}
Explanation: There are two distinct possible ambiguities, one is of type full_form which indicates query contains the string under full_form field (world boxng council in the query, not the WBC). The other one is of type abbreviation which indicates query contains the string under abbreviaiton field (US in the query not the United States)."""

user_message = """Query: {query}

Format Instructions:
{format_instructions}

Output:"""


output_parser = PydanticOutputParser(pydantic_object=QueryAmbiguation)
output_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)

messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message,
            input_variables=[],
            partial_variables={},
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=user_message,
            input_variables=["query"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | output_parser


def process_row(row: pd.Series):
    response: QueryAmbiguation = chain.invoke(
        {"query": row["question"]}, config=RunnableConfig(configurable={"llm": "gpt4"})
    )
    if response.full_form_abbrv_map:
        return response.json()


def extract_ambiguities(df: pd.DataFrame):
    with ThreadPoolExecutor(max_workers=3) as executor:
        df["possible_ambiguities"] = tqdm(executor.map(lambda x: process_row(x[1]), df.iterrows()), total=len(df))

    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)
    for i in tqdm(range(len(df))):
        question: str = df.loc[i, "question"]
        ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))
        unambiguous_question, ambiguous_question = question, question
        pops = []
        for i, amb in enumerate(ambiguities.full_form_abbrv_map):
            if amb.ambiguity_type == "abbreviation" and amb.abbreviation in question:
                unambiguous_question = unambiguous_question.replace(amb.abbreviation, amb.full_form)
                ambiguous_question = ambiguous_question
            elif amb.ambiguity_type == "full_form" and amb.full_form in question:
                unambiguous_question = unambiguous_question
                ambiguous_question = ambiguous_question.replace(amb.full_form, amb.abbreviation)
            else:
                pops.append(i)
        if ambiguous_question == unambiguous_question:
            continue

        for pop in reversed(pops):
            ambiguities.full_form_abbrv_map.pop(pop)
        df.loc[i, "possible_ambiguities"] = ambiguities.json() if ambiguities.full_form_abbrv_map else None
        df.loc[i, "ambiguous_question"] = ambiguous_question
        df.loc[i, "unambiguous_question"] = unambiguous_question
    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)
