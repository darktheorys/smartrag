import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from tqdm import tqdm

from llm import llm
from models import QueryAmbiguation

sys_message = """Given a query from a multi-hop complex question-answer dataset, your task is to identify full-forms or abbreviations contained in the query.
If query contains such full-form and abbreviation pairs, you will produce output accordingly.

In other words, if query contains a full-form that has a corresponding abbreviation or if query contains an abbreviation that has a corresponding full-form, you need to label it correct and extract necessary fields.

- Extract everything as is, without changing a single thing.

Example 1:
Query: Did Jack Dempsey fight the current WBC heavyweight champion?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "abbreviation", "abbreviation": "WBC", "full_form": "World Boxing Council"}}]}}

Example 2:
Query: Did Jack Dempsey fight the current World Boxing Council heavyweight champion?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "full_form", "abbreviation": "WBC", "full_form": "World Boxing Council"}}]}}

Example 3:
Query: Did Jack Dempsey fight the current world boxng council heavyweight champion in the US?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "full_form", "abbreviation": "WBC", "full_form": "world boxng council"}}, {{"ambiguity_type": "abbreviation", "abbreviation": "US", "full_form": "United States"}}]}}

{format_instructions}
"""

user_message = """Query: {query}
Output:"""


output_parser = PydanticOutputParser(pydantic_object=QueryAmbiguation)

messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message,
            input_variables=[],
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=user_message, input_variables=["query"], partial_variables={})
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | output_parser


def process_row(row: pd.Series):
    question = row["question"]

    response: QueryAmbiguation = chain.invoke({"query": question})
    if response.full_form_abbrv_map:
        return response.json()


def extract_ambiguities(df: pd.DataFrame):
    with ThreadPoolExecutor(max_workers=3) as executor:
        df["possible_ambiguities"] = tqdm(executor.map(lambda x: process_row(x[1]), df.iterrows()), total=len(df))

    for i in tqdm(range(len(df))):
        question: str = df.loc[i, "question"]
        ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))
        unambiguous_question, ambiguous_question = question, question

        for amb in ambiguities.full_form_abbrv_map:
            if amb.ambiguity_type == "abbreviation":
                assert amb.abbreviation in question, question
                unambiguous_question = unambiguous_question.replace(amb.abbreviation, amb.full_form)
                ambiguous_question = ambiguous_question
            elif amb.ambiguity_type == "full_form":
                unambiguous_question = unambiguous_question
                assert amb.full_form in question
                ambiguous_question = ambiguous_question.replace(amb.full_form, amb.abbreviation)

        df.loc[i, "ambiguous_question"] = ambiguous_question
        df.loc[i, "unambiguous_question"] = unambiguous_question
