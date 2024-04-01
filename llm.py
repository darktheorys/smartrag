from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import secrets

gpt35 = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=secrets.get("OPENAI_API_KEY"),
    max_tokens=4096,
    temperature=0.2,
    model_kwargs={"response_format": {"type": "json_object"}},
)

llm = (
    ChatOpenAI(
        model="gpt-4-turbo-preview",
        api_key=secrets.get("OPENAI_API_KEY"),
        max_tokens=4096,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    .configurable_alternatives(ConfigurableField("llm", name="llm"), default_key="gpt4", gpt35=gpt35)
    .configurable_fields(temperature=ConfigurableField(id="temperature"))
)

embedder = OpenAIEmbeddings(api_key=secrets.get("OPENAI_API_KEY"), model="text-embedding-3-large")
