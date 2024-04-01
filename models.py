from langchain.pydantic_v1 import BaseModel, Field


# from https://colab.research.google.com/drive/1nPpMyjxGLjD16WlgJgRpNWIPXy0ThJrW#scrollTo=m7kh3NOtiVjM&uniqifier=3
class QueryAmbiguation(BaseModel):
    class Ambiguity(BaseModel):
        ambiguity_type: str = Field(description="Either abbreviation or full-form")
        abbreviation: str
        full_form: str

    full_form_abbrv_map: list[Ambiguity] = Field(description="List of Mapping between full-form and abbreviation.")




class AnswerStr(BaseModel):
    answer: str


class AnswerBool(BaseModel):
    answer: bool


class AnswerJudge(BaseModel):
    selection: int


class Queries(BaseModel):
    questions: list[tuple[int, str]]


class Selector(BaseModel):
    selection_id: int
