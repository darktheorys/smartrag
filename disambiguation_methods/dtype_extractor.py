import pandas as pd
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from tqdm import tqdm

from llm import llm

dtypes = [
    (
        "Simple Datatype",
        """- Boolean: Questions that require a straightforward "yes" or "no" answer or a binary response.
    - Integer: Questions that expect numerical values without decimal points or fractional components.
    - Float: Questions that require numerical values with decimal points or fractional components.
    - String: Questions that call for textual responses or character-based information.""",
    ),
    (
        "Compound Datatype",
        """- List/Array: Questions that necessitate responses containing multiple items or a collection of values.
    - Dictionary/Map: Questions that require structured responses with key-value pairs or associative data.
    - Object/Structure: Questions that expect responses formatted as complex data structures or objects with attributes and properties.
    - Multimedia: Queries involving audio, video, or other multimedia formats.
    - and any combination of above.""",
    ),
]


class DtypeExtraction(BaseModel):
    selection: int = Field(description="Corresponding dtype index")
    subtype: str = Field(description="Name of the actual expected type")


sys_message = """You are an AI assistant. Follow the guidelines and requirements described carefully and then perform the given task using the queries from the user. 

    Task Guideline:data type in response to a given query by the user. 
    Pre-defined data types are provided below with their sub-type within that field. 
    Select only the data type domain from the corresponding list.

    Classes:
    {dtypes}

    {format_instructions}
"""

user_message = """Query: {query}"""

serialized_dtypes = "\n".join([f"{i} - {dt}\n\t" + sub for i, (dt, sub) in enumerate(dtypes)])


output_parser = PydanticOutputParser(pydantic_object=DtypeExtraction)
output_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)


messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message,
            input_variables=[],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions(),
                "dtypes": serialized_dtypes,
            },
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=user_message, input_variables=["query"], partial_variables={})
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | output_parser


def extract_dtypes(df: pd.DataFrame, llm: str = "gpt35", temp: float = 0) -> None:
    with tqdm(range(len(df))) as pbar:
        for i in pbar:
            query = df.loc[i, "ambiguous_question"] if "ambiguous_question" in df else df.loc[i, "question"]

            response: DtypeExtraction = chain.invoke(
                {"query": query},
                config=RunnableConfig(configurable={"llm": llm, "temperature": temp}),
            )

            dtype = dtypes[response.selection][0] if response.selection < len(dtypes) else ""
            pbar.set_postfix_str(f"Dtype: {dtype if dtype else None}")
            df.loc[i, "dtype"] = dtype + "-" + response.subtype
