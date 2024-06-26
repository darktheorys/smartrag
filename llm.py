from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import secrets

gpt35 = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=secrets.get("OPENAI_API_KEY"),
    max_tokens=4096,
    temperature=0.2,
    timeout=20,
    max_retries=5,
    model_kwargs={"response_format": {"type": "json_object"}},
)

llm = (
    ChatOpenAI(
        model="gpt-4-turbo",
        api_key=secrets.get("OPENAI_API_KEY"),
        max_tokens=4096,
        temperature=0.2,
        timeout=20,
        max_retries=5,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    .configurable_alternatives(ConfigurableField("llm", name="llm"), default_key="gpt4", gpt35=gpt35)
    .configurable_fields(
        temperature=ConfigurableField(id="temperature"), model_kwargs=ConfigurableField(id="model_kwargs")
    )
)

embedder = OpenAIEmbeddings(api_key=secrets.get("OPENAI_API_KEY"), model="text-embedding-3-large")
