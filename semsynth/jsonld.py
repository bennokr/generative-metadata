from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, get_type_hints
import logging


class JSONLDMixin:
    """Easy JSON-LD dataclass trick with roundtripping of unknown fields."""

    def __post_init__(self):
        # storage for unknown/non-dataclass keys from the source JSON-LD
        if not hasattr(self, "_extra"):
            self._extra = {}
        self._build_aliases()

    def _build_aliases(self):
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

    def to_jsonld(self, with_context=True, include_extra=True):
        def enc(v):
            if isinstance(v, JSONLDMixin):
                # nested JSONLDMixin instances still emit their extra keys
                return v.to_jsonld(with_context=False)
            if isinstance(v, (list, tuple)):
                return [enc(x) for x in v]
            return v

        doc = {}
        if with_context:
            ctx = getattr(self, "__context__", {})
            if ctx:
                doc["@context"] = ctx

        for f in self.fields_subclass_first():
            v = getattr(self, f.name)
            if v is not None:
                doc[f.name] = enc(v)

        if include_extra:
            extra = getattr(self, "_extra", {})
            for k, v in extra.items():
                # dataclass fields win if there is a collision
                if k not in doc:
                    doc[k] = enc(v)

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

            # Handle nested dataclasses that also use JSONLDMixin
            if isinstance(value, dict) and is_dataclass(ftype) and issubclass(ftype, JSONLDMixin):
                return ftype.from_jsonld(value)

            return value

        field_names = {f.name for f in fields(cls)}
        kwargs = {}

        # decode only known dataclass fields
        for f in fields(cls):
            if f.name in data:
                ftype = hints.get(f.name)
                if ftype is not None:
                    kwargs[f.name] = dec(data[f.name], ftype)
                else:
                    kwargs[f.name] = data[f.name]

        try:
            obj = cls(**kwargs)
        except Exception as e:
            logging.error(f"Invalid data with keys {list(data)}")
            raise e

        # capture extra keys (non-dataclass, non-@context)
        extra = {
            k: v
            for k, v in data.items()
            if k not in field_names and k != "@context"
        }
        if not hasattr(obj, "_extra"):
            obj._extra = {}
        obj._extra.update(extra)

        # merge class-level __context__ with incoming @context
        incoming_ctx = data.get("@context")
        class_ctx = getattr(obj, "__context__", {})

        merged_ctx = class_ctx
        if isinstance(class_ctx, dict) and isinstance(incoming_ctx, dict):
            # incoming context overrides class-level entries on conflict
            merged_ctx = {**class_ctx, **incoming_ctx}
        elif incoming_ctx is not None:
            # if incoming context is not a dict, just keep it
            merged_ctx = incoming_ctx

        if merged_ctx:
            # instance-level context shadowing the class-level one
            obj.__context__ = merged_ctx
            # rebuild aliases based on merged context
            obj._build_aliases()

        return obj
