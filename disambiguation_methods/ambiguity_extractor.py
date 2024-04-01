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
If query contains such full-form and abbreviation pairs, you will produce output accordingly. There can be more than one abbreviation/full-form inside given query. 
The output should look like a list if that is the case.

Extract everything as is, without changing a single thing, case, punctuation, encoding, anything.
After getting ambiguities from you, user will replace abbreviations with full-forms, therefore make sure everything make sense extracting the abbreviations and full-forms.
If you believe that the abbreviation-full form pairs create an ambiguity when answering the question, then you extract that.
If you dont think existing abbreviation-full form pairs are easy to misinterpret, then ignore that and continue.

Bad Example:
Query: What is (are) Arts syndrome ?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "abbreviation", "abbreviation": "Arts", "full_form": "Arts syndrome"}}]}}

Very Bad Example:
Query: What is (are) X-linked creatine deficiency ?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "abbreviation", "abbreviation": "X-linked", "full_form": "(are) X-linked creatine deficiency"}}]}}

Good Example:
Query: Did Jack Dempsey fight the current WBC heavyweight champion?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "abbreviation", "abbreviation": "WBC", "full_form": "World Boxing Council"}}]}}

Good Example:
Query: Did Jack Dempsey fight the current World Boxing Council heavyweight champion?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "full_form", "abbreviation": "WBC", "full_form": "World Boxing Council"}}]}}

Good Example:
Query: Did Jack Dempsey fight the current world boxng council heavyweight champion in the US?
Ambiguities: {{"full_form_abbrv_map": [{{"ambiguity_type": "full_form", "abbreviation": "WBC", "full_form": "world boxng council"}}, {{"ambiguity_type": "abbreviation", "abbreviation": "US", "full_form": "United States"}}]}}
"""

user_message = """Query: {query}

Format Instructions:
{format_instructions}

Output:"""


output_parser = PydanticOutputParser(pydantic_object=QueryAmbiguation)

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
    response: QueryAmbiguation = chain.invoke({"query": row["question"]})
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

        for amb in ambiguities.full_form_abbrv_map:
            if amb.ambiguity_type == "abbreviation" and amb.abbreviation in question:
                unambiguous_question = unambiguous_question.replace(amb.abbreviation, amb.full_form)
                ambiguous_question = ambiguous_question
            elif amb.ambiguity_type == "full_form" and amb.full_form in question:
                unambiguous_question = unambiguous_question
                ambiguous_question = ambiguous_question.replace(amb.full_form, amb.abbreviation)
        if ambiguous_question == unambiguous_question:
            continue
        df.loc[i, "ambiguous_question"] = ambiguous_question
        df.loc[i, "unambiguous_question"] = unambiguous_question
    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)
