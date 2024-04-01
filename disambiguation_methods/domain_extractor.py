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

categories = [
    (
        "SCIENCE",
        (
            "Amateur Radio",
            "Architecture",
            "Biology",
            "Chemistry",
            "Degrees",
            "Electronics",
            "Geology",
            "IEEE",
            "Mathematics",
            "Mechanics",
            "Meteorology",
            "Ocean Science",
            "Physics",
            "Universities",
        ),
    ),
    (
        "COMMUNITY",
        (
            "Conferences",
            "Educational",
            "Famous",
            "Film Censorship",
            "Genealogy",
            "Housing",
            "Law",
            "Media",
            "Museums",
            "Music",
            " Non-Profit Organizations",
            "Religion",
            "Schools",
            "Sports",
            "Unions",
        ),
    ),
    (
        "BUSINESS",
        (
            "Accounting",
            "Firms",
            "International Business",
            "Mortgage",
            "NASDAQ Symbols",
            "NYSE Symbols",
            "Occupations & Positions",
            "Professional Organizations",
            "Stock Exchange",
            "Tax",
        ),
    ),
    (
        "GOVERNMENTAL",
        ("FBI", "FDA", "Military", "NASA", "Police", "State & Local", "Suppliers", "Transportation", "UN", "US Gov"),
    ),
    (
        "INTERNET",
        ("ASCII", "Blogs", "Chat", "Domain Names", "Emoticons", "HTTP", "MIME", "Twitter", "Wannas", "Websites"),
    ),
    (
        "MISCELLANEOUS",
        (
            "Chess",
            "Clothes",
            "Coins",
            "Construction",
            "Days",
            "Farming",
            "Food",
            "Funnies",
            "Gaming",
            "Hobbies",
            "Months",
            "Photography",
            "Plastics",
            "Sci-Fi",
            "Unit Measures",
            "Journal Abbreviations",
        ),
    ),
    (
        "REGIONAL",
        (
            "Airport Codes",
            "African",
            "Alaska",
            "Australian",
            "Canadian",
            "Cities",
            "Countries",
            "Currencies",
            "European",
            "Language Codes",
            "Railroads",
            "Tel. Country Codes",
            "Time Zones",
            "US States",
        ),
    ),
    (
        "MEDICAL",
        (
            "British Medicine",
            "Dental",
            "Drugs",
            "Hospitals",
            "Human Genome",
            "Laboratory",
            "Medical Physics",
            "Neurology",
            "Nursing",
            "Oncology",
            "Physiology",
            "Prescription",
            "Veterinary",
        ),
    ),
    (
        "INTERNATIONAL",
        (
            "Arabic",
            "Dutch",
            "German",
            "Greek",
            "Guatemalan",
            "French",
            "Hebrew",
            "Indonesian",
            "Italian",
            "Latin",
            "Mexican",
            "Polish",
            "Romanian",
            "Russian",
            "Spanish",
            "Tamil",
            "Thai",
            "Turkish",
        ),
    ),
]


class DomainExtraction(BaseModel):
    selection: int = Field(description="Corresponding domain index")


sys_message = """User will provide you with a query. Your task is to classify that query into corresponding domain name. 
Domain names are provided below with their subset of subcategories within that domain. Select only one domain from corresponding list.

{categories}

{format_instructions}
"""

user_message = """Query: {query}
Output:"""

serialized_categories = "\n".join([f"{i} - {cat}\n\t-" + "\n\t-".join(ex) for i, (cat, ex) in enumerate(categories)])


output_parser = PydanticOutputParser(pydantic_object=DomainExtraction)

messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message,
            input_variables=[],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions(),
                "categories": serialized_categories,
            },
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=user_message, input_variables=["query"], partial_variables={})
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | output_parser


def extract_domains(df: pd.DataFrame) -> None:
    for i in tqdm(range(len(df))):
        query = df.loc[i, "ambiguous_question"]

        response: DomainExtraction = chain.invoke({"query": query})

        df.loc[i, "domain"] = categories[response.selection][0] if response.selection < len(categories) else None
