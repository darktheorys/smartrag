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

categories = [
    (
        "SCIENCE",
        "Academic & Science",
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
        "COMPUTING",
        "Computing",
        (
            "Assembly",
            "Databases",
            "DOS Commands",
            "Drivers",
            "File Extensions",
            "General",
            "Hardware",
            "Java",
            "Networking",
            "Security",
            "Software",
            "Telecom",
            "Texting",
            "Unix Commands",
        ),
    ),
    (
        "COMMUNITY",
        "Community",
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
        "Business & Finance",
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
        "Governmental",
        ("FBI", "FDA", "Military", "NASA", "Police", "State & Local", "Suppliers", "Transportation", "UN", "US Gov"),
    ),
    (
        "INTERNET",
        "Internet",
        ("ASCII", "Blogs", "Chat", "Domain Names", "Emoticons", "HTTP", "MIME", "Twitter", "Wannas", "Websites"),
    ),
    (
        "MISCELLANEOUS",
        "Miscellaneous",
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
        "Regional",
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
        "Medical",
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
        "International",
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

serialized_categories = "\n".join(
    [f"{i} - {cat}\n\t-" + "\n\t-".join(ex) for i, (cat, alias, ex) in enumerate(categories)]
)


output_parser = PydanticOutputParser(pydantic_object=DomainExtraction)
output_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)


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


def extract_domains(df: pd.DataFrame, llm: str = "gpt35", temp: float = 0) -> None:
    with tqdm(range(len(df))) as pbar:
        for i in pbar:
            query = df.loc[i, "ambiguous_question"]

            response: DomainExtraction = chain.invoke(
                {"query": query}, config=RunnableConfig(configurable={"llm": llm, "temperature": temp})
            )

            df.loc[i, "domain_idx"] = response.selection
            category = categories[response.selection] if response.selection < len(categories) else None
            pbar.set_postfix_str(f"Domain: {category[0] if category else None}")
        df["domain_idx"] = df["domain_idx"].astype(int)
