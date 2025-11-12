from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, get_type_hints
import logging

class Namespace(str):
    """Easy IRI Namespace trick"""

    def __getattr__(self, key):
        # Avoid hijacking Python special/dunder lookups (e.g., __deepcopy__)
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        return Namespace(self + key)

    def __getitem__(self, key):
        return self + key


class JSONLDMixin:
    """Easy JSON-LD dataclass trick

    >>> @dataclass
    >>> class Book(JSONLDMixin):
    >>>     __context__ = {
    >>>         "dct": "http://purl.org/dc/terms/",
    >>>         "title": "dct:title",
    >>>         "creator": "dct:creator",
    >>>     }
    >>>     title: str
    >>>     creator: str
    >>> b = Book("The Hobbit", "J. R. R. Tolkien")
    >>> assert b["title"] == b["dct:title"] == b.title
    >>> print(b.to_jsonld())
    {
      '@context': {'dct': 'http://purl.org/dc/terms/', 'title': 'dct:title', 'creator': 'dct:creator'},
      'title': 'The Hobbit',
      'creator': 'J. R. R. Tolkien'
    }
    """

    def __post_init__(self):
        self.__alias = {f.name: f.name for f in fields(self)}
        ctx = getattr(self, "__context__", {})
        for term, val in ctx.items():
            if term in self.__alias:
                if isinstance(val, dict) and "@id" in val:
                    val = val["@id"]
                if isinstance(val, str):
                    # Support QName (e.g., "dct:title")
                    if ":" in val and not val.startswith(("http://", "https://")):
                        self.__alias[val] = term
                        prefix, local = val.split(":", 1)
                        base = ctx.get(prefix)
                        if isinstance(base, str):
                            self.__alias[base + local] = term
                    # Support direct full URI in @id or context
                    elif val.startswith(("http://", "https://")):
                        self.__alias[val] = term

    def __getitem__(self, key):
        k = self.__alias.get(key)
        if not k:
            raise KeyError(key)
        return getattr(self, k)

    @classmethod
    def fields_subclass_first(cls):
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        fby = {f.name: f for f in fields(cls)}
        seen, out = set(), []
        for C in cls.__mro__:
            if C is object:
                break
            for n in C.__dict__.get("__annotations__", ()):
                if n in fby and n not in seen:
                    seen.add(n)
                    out.append(fby[n])
        return tuple(out)

    def to_jsonld(self, with_context=True):
        def enc(v):
            if isinstance(v, JSONLDMixin):
                return v.to_jsonld(with_context=False)
            if isinstance(v, (list, tuple)):
                return [enc(x) for x in v]
            return v

        doc = {"@context": getattr(self, "__context__", {})} if with_context else {}
        for f in self.fields_subclass_first():
            v = getattr(self, f.name)
            if v is not None:
                doc[f.name] = enc(v)
        return doc

    @classmethod
    def from_jsonld(cls, data: dict):
        hints = get_type_hints(cls)
        def dec(value, ftype):
            origin = get_origin(ftype)

            # Handle Optional[T] / Union[T, None]
            if origin is Union:
                args = get_args(ftype)
                non_none = [a for a in args if a is not type(None)]
                # Optional[T]
                if len(non_none) == 1:
                    if value is None:
                        return None
                    return dec(value, non_none[0])
                # more complex unions could be handled here as needed
                return value  # fallback

            # Handle list/tuple[T]
            if origin in (list, tuple):
                (subtype,) = get_args(ftype)
                return [dec(v, subtype) for v in value]

            # Handle nested dataclasses that also use JSONMixin
            if isinstance(value, dict) and is_dataclass(ftype) and issubclass(ftype, JSONLDMixin):
                return ftype.from_jsonld(value)

            return value

        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = dec(data[f.name], hints[f.name])

        try:
            return cls(**kwargs)
        except Exception as e:
            logging.error(f"Invalid data with keys {list(data)}")
            raise e
