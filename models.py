from langchain.pydantic_v1 import BaseModel, Field


# from https://colab.research.google.com/drive/1nPpMyjxGLjD16WlgJgRpNWIPXy0ThJrW#scrollTo=m7kh3NOtiVjM&uniqifier=3
class QueryAmbiguation(BaseModel):
    class Ambiguity(BaseModel):
        ambiguity_type: str = Field(
            description="Either 'abbreviation' or 'full_form', depending on the existence in the query."
        )
        abbreviation: str = Field(description="Abbreviation of the full_form. (e.g: US, WWE, www, ...)")
        full_form: str = Field(
            description="Full_form of the abbreviation. (e.g: United States, World Wrestling Entertainment, World wide web)"
        )

    full_form_abbrv_map: list[Ambiguity] = Field(
        description="List of Mapping between full-form and abbreviation pairs."
    )
