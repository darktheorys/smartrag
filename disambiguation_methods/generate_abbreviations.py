import pandas as pd
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from tqdm import tqdm

from llm import llm
from models import QueryAmbiguation


class AmbiguousQueryTuple(BaseModel):
    query_1: str = Field(description="First Ambiguous query generated from abbreviation")
    answer_1: str = Field(description="Answer to the first query")
    query_2: str = Field(description="Second Ambiguous query generated from abbreviation")
    answer_2: str = Field(description="Answer to the second query")


class Abbreviation(BaseModel):
    abbreviation: str
    full_form_1: str
    full_form_2: str


sys_message = """You are a helpful assistant. No yapping. Just do as you told. Do not interact or inform the user. Make sure to follow them or you will be shutdown."""

generate_abbrevations_stage_1 = """Find a single, real, ambiguous abbreviation, acronym that has at least two distinct full-forms and meanings, then provide two full-forms, it can be from Finance, Marketing, Technology, Science, Medical, Computing, Governmentak domains.

Create something different than the ones listed below:
{previous_abbrvs}

{format_instructions}
"""
generate_abbrevations_stage_2 = """Your aim is to generate a query tuple with the descriptions below.

Given two distinct terms: "{full_form_1}" and "{full_form_2}"

-Use those two distinct terms to create two separate queries and aim for a different/distinct specific answer depending on them.
-Generated query tuples should aim for different answers, but their domain can be same.
-Queries can be multi-hop, complex, hard to answer and retrieval enabling. Additionally, possible answer to those queries should depend on the full-form of abbreviation, meaning it should not be an expression rather concept.
-Queries should definitely contain the terms given above, as is (exactly the with the given form above), not the abbreviations or other names for them.
-Finally, try to hide the focus on the abbreviation, make queries natural and close to real-life scenarios.
-Do not produce very long sequences, being concise and following other rules are enough.

{format_instructions}
"""
abbrv_parser = PydanticOutputParser(pydantic_object=Abbreviation)
abbrv_parser = OutputFixingParser.from_llm(llm=llm, parser=abbrv_parser)

query_parser = PydanticOutputParser(pydantic_object=AmbiguousQueryTuple)
query_parser = OutputFixingParser.from_llm(llm=llm, parser=query_parser)

transform = RunnableLambda(lambda x: x.dict())


messages_abbrv = [
    SystemMessagePromptTemplate(prompt=PromptTemplate(template=sys_message, input_variables=[])),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=generate_abbrevations_stage_1,
            input_variables=["previous_abbrvs"],
            partial_variables={"format_instructions": abbrv_parser.get_format_instructions()},
        )
    ),
]
prompt_abbrv = ChatPromptTemplate.from_messages(messages=messages_abbrv)


messages_query = [
    SystemMessagePromptTemplate(prompt=PromptTemplate(template=sys_message, input_variables=[])),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=generate_abbrevations_stage_2,
            input_variables=["full_form_1", "full_form_2"],
            partial_variables={"format_instructions": query_parser.get_format_instructions()},
        )
    ),
]
prompt_query = ChatPromptTemplate.from_messages(messages=messages_query)

query_chain = prompt_query | llm | query_parser
abbrv_chain = prompt_abbrv | llm | abbrv_parser

chain = abbrv_chain | {
    "query_chain": transform | query_chain,
    "abbreviation_chain": RunnablePassthrough(),
}


def generate_ambiguous_queries(n_queries: int = 1, temperature: float = 1.0) -> pd.DataFrame:
    data = []
    previously_generated_abbreviations = set()

    with tqdm(total=n_queries) as pbar:
        while len(data) < 2 * n_queries:
            chain_result = chain.invoke(
                {"previous_abbrvs": "\n".join(previously_generated_abbreviations)},
                config=RunnableConfig(configurable={"temperature": temperature}),
            )
            abbrv: Abbreviation = chain_result["abbreviation_chain"]
            query_tuple: AmbiguousQueryTuple = chain_result["query_chain"]

            if abbrv.json() in previously_generated_abbreviations:
                continue

            previously_generated_abbreviations.add(abbrv.json())

            if abbrv.full_form_1 in query_tuple.query_1 and abbrv.full_form_2 in query_tuple.query_2:
                df_data = [
                    {
                        "possible_ambiguities": QueryAmbiguation(
                            full_form_abbrv_map=[
                                QueryAmbiguation.Ambiguity(
                                    ambiguity_type="abbreviation",
                                    abbreviation=abbrv.abbreviation,
                                    full_form=abbrv.full_form_1,
                                )
                            ]
                        ).json(),
                        "question": query_tuple.query_1.replace(abbrv.full_form_1, abbrv.abbreviation),
                        "ambiguous_question": query_tuple.query_1.replace(abbrv.full_form_1, abbrv.abbreviation),
                        "unambiguous_question": query_tuple.query_1,
                        "answer": query_tuple.answer_1,
                    },
                    {
                        "possible_ambiguities": QueryAmbiguation(
                            full_form_abbrv_map=[
                                QueryAmbiguation.Ambiguity(
                                    ambiguity_type="abbreviation",
                                    abbreviation=abbrv.abbreviation,
                                    full_form=abbrv.full_form_2,
                                )
                            ]
                        ).json(),
                        "question": query_tuple.query_2.replace(abbrv.full_form_2, abbrv.abbreviation),
                        "ambiguous_question": query_tuple.query_2.replace(abbrv.full_form_2, abbrv.abbreviation),
                        "unambiguous_question": query_tuple.query_2,
                        "answer": query_tuple.answer_2,
                    },
                ]
                data.extend(df_data)
                pbar.update()
                pbar.refresh()
        return pd.DataFrame(data=data)
