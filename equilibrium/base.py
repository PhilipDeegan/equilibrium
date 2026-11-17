import pydantic


class BaseModel(pydantic.BaseModel):
    def _set_(self, **kwargs):
        for k in self.__class__.__fields__.keys():
            if v := kwargs.get(k):
                getattr(self, k)(v)
