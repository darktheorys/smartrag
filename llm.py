from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

from utils import secrets

llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    api_key=secrets.get("OPENAI_API_KEY"),
    max_tokens=4096,
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}},
).configurable_fields(temperature=ConfigurableField(id="temperature"))
