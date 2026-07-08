import configparser
import logging
from dataclasses import fields
from typing import get_args, get_origin

logger = logging.getLogger(__name__)


def parse_overrides(overrides: list[str]) -> dict[str, str]:
    """Parse CLI overrides of the form key=value."""
    result = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_value(raw: str, field_type):
    """Parse a string value into the appropriate Python type."""
    origin = get_origin(field_type)

    # Handle Optional types (Union[X, None])
    if origin is type(None):
        return None
    # str | None
    if hasattr(field_type, "__args__") and type(None) in getattr(
        field_type, "__args__", ()
    ):
        non_none = [a for a in field_type.__args__ if a is not type(None)]
        if raw.lower() == "none":
            return None
        if non_none:
            return _parse_value(raw, non_none[0])
        return raw

    # Literal
    if origin is type(None):
        return None
    try:
        from typing import Literal

        if get_origin(field_type) is Literal:
            allowed = get_args(field_type)
            if raw not in [str(a) for a in allowed]:
                raise ValueError(
                    f"Value '{raw}' not in allowed values: {allowed}"
                )
            # Return the correct type
            for a in allowed:
                if str(a) == raw:
                    return a
            return raw
    except ImportError:
        pass

    if field_type is bool:
        return raw.lower() in ("true", "1", "yes")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is str:
        return raw

    # list[str]
    if origin is list:
        return [item.strip() for item in raw.split(",")]

    return raw


def config_to_dataclass(config_path: str, overrides: list[str], dataclass_type):
    """Load a .cfg file and construct a dataclass instance.

    Fields are populated from the config file, with CLI overrides taking precedence.
    Missing fields fall back to dataclass defaults.
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)
    read_files = parser.read(config_path)
    
    if not parser.has_section("config"):
        raise ValueError(f"Config file {config_path} must have a [config] section")

    override_dict = parse_overrides(overrides)
    config_dict = dict(parser["config"])

    kwargs = {}
    for f in fields(dataclass_type):
        raw = None
        if f.name in override_dict:
            raw = override_dict[f.name]
            logger.info(f"Override: {f.name} = {raw}")
        elif f.name in config_dict:
            raw = config_dict[f.name]

        if raw is not None:
            kwargs[f.name] = _parse_value(raw, f.type)

    return dataclass_type(**kwargs)
