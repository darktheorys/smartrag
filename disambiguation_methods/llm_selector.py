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
from tqdm import tqdm

from disambiguation_methods.domain_extractor import categories
from llm import llm
from models import QueryAmbiguation


class Selector(BaseModel):
    selection_id: int


selector_output_parser = PydanticOutputParser(pydantic_object=Selector)
selector_output_parser = OutputFixingParser.from_llm(llm=llm, parser=selector_output_parser)

selector_system_prompt = """User will give you a query, in this query there will be an abbreviation. Your task is to resolve that abbreviation.
User will also provide possible full-forms that you can select from. Please do best you can while selecting from the given list of options.
If you cant find and appropriate selection from given options, please use selection_id as -1.

Query domain will be {domain}.

Format Instructions:
{format_instructions}
"""

selector_user_message = """Query: {query}
Abbreviation: {abbrv}
Options:
{options}
Selection:"""

selector_messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=selector_system_prompt,
            input_variables=["domain"],
            partial_variables={
                "format_instructions": selector_output_parser.get_format_instructions(),
            },
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=selector_user_message, input_variables=["query", "abbrv", "options"])
    ),
]
selector_prompt = ChatPromptTemplate.from_messages(messages=selector_messages)
selector_chain = selector_prompt | llm | selector_output_parser


def get_llm_selections(df: pd.DataFrame, top_n: int):
    for query_n in tqdm(range(len(df))):
        ambiguous_question = df.loc[query_n, "ambiguous_question"]
        ambiguities = QueryAmbiguation(**json.loads(df.loc[query_n, "possible_ambiguities"]))

        all_ambiguity_suggestions: list[list[str]] = [[] for _ in ambiguities.full_form_abbrv_map]
        # if there is a suggestion from APIs
        if isinstance(df.loc[query_n, f"top_{top_n}_full_form"], str):
            api_suggestions: list[list[str]] = json.loads(df.loc[query_n, f"top_{top_n}_full_form"])
            for i in range(len(api_suggestions)):
                all_ambiguity_suggestions[i] += api_suggestions[i]
            # add llm suggestion to df

        most_likely_full_forms = []
        most_likely_selection_type = []
        for idx, (suggestions, amb, sources) in enumerate(
            zip(
                all_ambiguity_suggestions,
                ambiguities.full_form_abbrv_map,
                json.loads(df.loc[query_n, f"top_{top_n}_full_form_sources"]),
            )
        ):
            resp: Selector = selector_chain.invoke(
                {
                    "abbrv": amb.abbreviation,
                    "query": ambiguous_question,
                    "options": [f"{i} - {opt}\n" for i, opt in enumerate(suggestions)],
                    "domain": categories[df.loc[i, "domain_idx"]][0]
                    if df.loc[i, "domain_idx"] < len(categories)
                    else None,
                }
            )
            most_likely_full_forms.append(
                suggestions[resp.selection_id]
                if resp.selection_id != -1
                else json.loads(df.loc[query_n, "llm_full_form_suggestions"])[idx]
            )
            most_likely_selection_type.append(sources[resp.selection_id] if resp.selection_id != -1 else "LLM")
        df.loc[query_n, "LLM_most_likely_full_forms"] = json.dumps(most_likely_full_forms)
        df.loc[query_n, "LLM_most_likely_selection_types"] = json.dumps(most_likely_selection_type)
