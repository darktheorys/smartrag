from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig

from llm import llm

prompt = """Your task is to create a strategy to answer a specific query using the guidelines below. 

Here is the guidelines for creating the strategy accordingly, you need to classify question for each of the dimensions:
- Datatype (Structure_Definition):
    1. Simple Datatypes:
        - Boolean: Questions that require a straightforward binary response like yes/no, or true/false.
        - Integer: Questions that expect numerical values without decimal points or fractional components, like classification.
        - Float: Questions that require numerical values with decimal points or fractional components.
        - String: Questions that call for textual responses or character-based information.
    2. Compound Datatypes:
        - List/Array: Questions that necessitate responses containing multiple items or a collection of values.
        - Dictionary/Map: Questions that require structured responses with key-value pairs or associative data.
        - Multimedia: Queries involving audio, video, or other multimedia formats.

- Contextual Sensitivity (Context_Classification):
    1. Context Independent: These questions typically seek information about general concepts or fundamental principles that remain consistent over time/context. The context in which they're asked may change, but the core concepts remain unchanged.
    2. Context Dependent: Questions falling under this category involve topics subject to periodic changes or updates according the context they are in, yet the core elements of the answer remain stable. The context may shift over time, requiring occasional updates to reflect evolving standards or practices.
    3. Context Sensitive: These questions pertain to topics experiencing frequent or unpredictable changes with respect to different contexts, demanding continuous updates to the answer to keep pace with shifting conditions or emerging trends. The context surrounding these questions is dynamic and requires ongoing monitoring.

- Query Understanding Classification (Domain_Extraction and/or Disambiguation):
    1. Explicit: This category is reserved for queries that are articulated with precise language and directly reference specific concepts or entities relevant to the user's query. The queries under this classification convey the information request in a straightforward manner, enabling a direct response without the need for inferring context or deciphering ambiguous references. The focus is on the clarity and directness of the query's phrasing, ensuring immediate understanding without extrapolation.
    2. Implicit: Implicit queries are less clear or direct, requiring interpretation of the context to provide a relevant response. This classification is applied to queries that are not directly articulated, often involving references to unknown concepts or entities that are not explicitly mentioned. Such queries require the Large Language Model to infer the missing pieces or to understand the context of the query without direct guidance from the user. The challenge lies in identifying and understanding these elements that are implied rather than clearly stated, to fulfill the query's requirements.

- Retrieval & Synthesis Dimension (Retrieval):
    1. No Retrieval Inquiry: Interactions where the LLM relies solely on its internal knowledge base or generative capabilities to respond to queries. This involves situations where the LLM does not need to access or retrieve information from specific data sources or synthesize across different pieces of information. Instead, it can generate responses based on the patterns, concepts, and information it has learned during its training phase. This could involve generating creative content, predicting outcomes based on learned patterns, or applying generalized knowledge to hypothesize answers for hypothetical scenarios. This category represents the model's ability to utilize learned information without the explicit need for current, real-world data retrieval. These interactions showcase the model's capacity for abstraction, creativity, and application of general knowledge in generating responses.
    2. Simple Retrieval Inquiry: Queries involving straightforward retrieval of information without complex search or processing requirements. Simple Retrieval Inquiries involve accessing information that can be readily retrieved from one or more sources without processing. These inquiries typically pertain to factual information or well-defined data points that are easily accessible. Information can be readily retrieved from one or more sources. (think of it as a puzzle pieces, that can fit each other without any effort) This involves instances where the LLM fetches and delivers information directly as it is stored in its data sources, without needing to reframe, interpret, or significantly process the information beyond recognizing the inquiry's context. This is akin to quoting facts or data directly from the dataset. Such inquiries do not require the model to understand or synthesize information across contexts — it is essentially retrieving and presenting information as-is.
    3. Synthetic Retrieval Inquiry: Queries requiring advanced search techniques, multiple sequential steps, or processing to retrieve relevant information effectively. Synthetic Retrieval Inquiries involve accessing information that requires synthesis or integration of data from multiple sources or perspectives to generate a comprehensive response. These inquiries often involve complex concepts, interrelated data points, or nuanced interpretations that cannot be easily obtained from a single source. Synthesis of information to provide a comprehensive response. (think of it as wood, to make a table you need to process it) This category applies when the LLM must contextually process information from different segments or times, integrate diverse information or data sources, or apply reasoning to generate a response. This goes beyond mere retrieval, involving understanding contexts, drawing inferences, or synthesizing information in ways that werent explicitly outlined in the source material. This can include cross-referencing facts, identifying relationships between concepts, or integrating knowledge from various contexts to develop a comprehensive, coherent response.
    
- Query Intent (Intent_Extraction):
    1. Informational/Instructional: Encompasses seeking factual information, definitions, and explanations. This category is for queries that aim directly at understanding specific facts or data.
    2. Analytical/Evaluative(Convergent Thinking): This merged category is for queries that involve dissecting complex information or data to understand underlying patterns, reasons, or to make informed judgements. Queries in this category seek a deeper understanding that supports decision-making, comparison of options, or evaluating alternatives. The process involves both analysis (to break down and understand) and evaluation (to assess, compare, and decide), acknowledging that these processes often occur in tandem. This category involves queries where the goal is to converge upon a specific answer or decision through analysis and evaluation.
    3. Exploratory(Divergent Thinking): Queries that are inherently about seeking new areas of knowledge, understanding emerging trends, or identifying unknown opportunities without a specific end goal of making a decision or judgment. Exploratory queries are characterized by their openness and the lack of a predefined objective, differentiating them from evaluative queries which are directed towards forming a judgment or assessment. Exploratory queries encourage branching out into various directions to explore a wide array of possibilities without necessarily aiming to converge on a single answer or outcome.
    4. Generative: Generative queries should be distinctly categorized to emphasize their creative and output-generating nature. Unlike exploratory queries that ***diverge*** in the search for new knowledge or trends, generative queries specifically seek the creation of new content, ideas, or solutions. Generative queries implicate a divergent thought process focused on originality and creation, clearly setting it apart from exploratory intent.
    5. Composite: This category includes queries that do not fit neatly into the above categories or have unique/multiple intents not covered by the previous categories.

After classifying the query, you need to plan a strategy to answer the question. It may or may not include the findings from the classification.
Indicate what actions you need in each step explicitly in imperative form like a checklist, which will be processed sequentially to fullfill the strategy.
For each of the steps, give a brief title related with the related classifications above, and also put the classifications into findings section.
Also, create a graph using the operations to follow from source to the final answer, including what data to transfer between nodes.

{format_instructions}

"""


class RetrievalStrategy(BaseModel):
    steps: list[str]
    findings: list[str]
    graph: dict


strategy_parser = OutputFixingParser.from_llm(
    llm=llm.with_config(config=RunnableConfig(configurable={"llm": "gpt35"})),
    parser=PydanticOutputParser(pydantic_object=RetrievalStrategy),
)

user_message = """Query:{query}"""

messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=prompt,
            input_variables=[],
            partial_variables={
                "format_instructions": strategy_parser.get_format_instructions(),
            },
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=user_message, input_variables=["query"], partial_variables={})
    ),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | llm | strategy_parser
