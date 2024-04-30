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
        "This domain represents the academic and scientific fields. It includes subcategories like Biology, Chemistry, Mathematics, and Physics.",
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
        "This domain represents the computing field from hardware to software. It includes subcategories like Databases, Networking, and Software.",
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
        "This domain represents the community field, as the name suggests. It includes subcategories like Educational, Law, and Music.",
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
        "This domain represents the business and finance fields with subcategories like Accounting, Tax, and Stock Exchange.",
    ),
    (
        "GOVERNMENTAL",
        "Governmental",
        ("FBI", "FDA", "Military", "NASA", "Police", "State & Local", "Suppliers", "Transportation", "UN", "US Gov"),
        "This domain represents all the governmental organizations and departments. It includes subcategories like FBI, NASA, and US",
    ),
    (
        "INTERNET",
        "Internet",
        ("ASCII", "Blogs", "Chat", "Domain Names", "Emoticons", "HTTP", "MIME", "Twitter", "Wannas", "Websites"),
        "This domain represents the internet field with subcategories like Blogs, Chat, and Websites.",
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
        "This domain represents miscellaneous topics that do not fit into the above categories. It includes subcategories like Chess, Food, and Sci-Fi."
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
        "This domain represents regional topics like countries, cities, and languages. It includes subcategories like African, European, and US States."
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
        "This domain represents the medical field with subcategories like Dental, Drugs, and Neurology. It is the one that deals with health, medicine, viruses, and diseases."
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
        "This domain represents international topics and languages. It includes subcategories like Arabic, French, and Spanish."
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
            query = df.loc[i, "ambiguous_question"] if "ambiguous_question" in df else df.loc[i, "question"]

            response: DomainExtraction = chain.invoke(
                {"query": query}, config=RunnableConfig(configurable={"llm": llm, "temperature": temp})
            )

            df.loc[i, "domain_idx"] = response.selection
            category = categories[response.selection][0] if response.selection < len(categories) else None
            pbar.set_postfix_str(f"Domain: {category if category else None}")
            df.loc[i, "domain"] = category
        df["domain_idx"] = df["domain_idx"].astype(int)
