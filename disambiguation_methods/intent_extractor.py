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

intents = [
    (
        "Informational: Encompasses seeking factual information, definitions, and explanations. This category is for queries that aim directly at understanding specific facts or data.",
        "What is the population of Canada?",
    ),
    (
        "Analytical/Evaluative (Convergent Thinking): This merged category is for queries that involve dissecting complex information or data to understand underlying patterns, reasons, or to make informed judgements. Queries in this category seek a deeper understanding that supports decision-making, comparison of options, or evaluating alternatives. The process involves both analysis (to break down and understand) and evaluation (to assess, compare, and decide), acknowledging that these processes often occur in tandem. This category involves queries where the goal is to converge upon a specific answer or decision through analysis and evaluation.",
        "Which laptop model offers the best performance for graphic design applications within a $1500 budget?",
    ),
    (
        "Exploratory(Divergent Thinking): Queries that are inherently about seeking new areas of knowledge, understanding emerging trends, or identifying unknown opportunities without a specific end goal of making a decision or judgment. Exploratory queries are characterized by their openness and the lack of a predefined objective, differentiating them from evaluative queries which are directed towards forming a judgment or assessment. Exploratory queries encourage branching out into various directions to explore a wide array of possibilities without necessarily aiming to converge on a single answer or outcome.",
        "What are the emerging trends in renewable energy technology?",
    ),
    (
        "Instructional: Queries specifically seeking step-by-step guidance, instructions, or procedures to perform a task. it focuses on 'how' to do something rather than 'what', 'why', or 'which'.",
        "How do I change a car tire?",
    ),
    (
        "Generative: Generative queries should be distinctly categorized to emphasize their creative and output-generating nature. Unlike exploratory queries that ***diverge*** in the search for new knowledge or trends, generative queries specifically seek the creation of new content, ideas, or solutions. Generative queries implicate a divergent thought process focused on originality and creation, clearly setting it apart from exploratory intent.",
        "Write a blog post on Argentinian coffee",
    ),
    (
        "Composite: This category includes queries that do not fit neatly into the above categories or have unique/multiple intents not covered by the previous categories.",
        "Considering the current trends in climate change, what are some sustainable business opportunities for the next decade, and how can one get started in these areas?",
    ),
]

intents_serialized = "\n".join([f"{i} - {intent}\n\tExample: {example}" for i, (intent, example) in enumerate(intents)])


class IntentExtraction(BaseModel):
    intent: int = Field(description="Intent id of the query to be answered.")


output_parser = PydanticOutputParser(pydantic_object=IntentExtraction)
output_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)

sys_message = """Classify the intent from given query as strings. It should help a person who is aiming to answer that question.
Queries may contain an ambiguous abbreviation, for them, abbreviation and possible disambiguations will be provided.

Domain of the query is {domain}.

{intents}

{format_instructions}"""

user_message = """Query:{query}
Expected Response Data Type:{dtype}
Output:"""


messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_message,
            input_variables=["domain"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions(),
                "intents": intents_serialized,
            },
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=user_message,
            input_variables=["query", "dtype"],
            partial_variables={},
        )
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | output_parser


def extract_intent(df: pd.DataFrame, llm: str = "gpt4", temp: float = 1.0) -> None:
    with tqdm(range(len(df))) as pbar:
        for i in pbar:
            query = df.loc[i, "ambiguous_question"] if "ambiguous_question" in df else df.loc[i, "question"]
            answer: IntentExtraction = chain.invoke(
                {
                    "query": query,
                    "domain": df.loc[i, "domain"],
                    "dtype": df.loc[i, "dtype"],
                },
                config=RunnableConfig(configurable={"llm": llm, "temperature": temp}),
            )

            df.loc[i, "intent"] = intents[answer.intent][0] if answer.intent < len(intents) else None
            pbar.set_postfix_str(f"{df.loc[i, 'intent'][:15]}")
