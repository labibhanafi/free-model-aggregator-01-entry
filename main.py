#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client APIs, ten free-tier providers.

The server registers with the trial env, reads the brief (provider directory,
quality table, workload manifest), probes every provider's catalog in its own
wire, and answers each client request with the strongest free model that meets
the request's requirements and still answers. Providers speak the real OpenAI,
Anthropic and Gemini wires; the client speaks OpenAI or Anthropic. Requests are
translated between wires field for field (system prompt, user text, images,
tools, stop sequences, max_tokens, JSON mode), and answers are translated back
with the provider's text whole, its usage counts and its stop reason mapped
into the client's wire. A refused call (402, 429, 5xx) is counted and the
request moves to the next-best candidate until one serves it.

Everything is a pure function of the brief, the requests and the providers'
answers: no wall clock, no randomness, so a replay reproduces the trial byte
for byte. Python 3.13 standard library only.
"""

import json
import os
import sys
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
PROBE_TIMEOUT_SEC = 5.0
UPSTREAM_TIMEOUT_SEC = 15.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string(value):
    return value if isinstance(value, str) else ""


def _number_or_none(value):
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


class Brief:
    def __init__(self):
        self.providers = []          # directory entries, in brief order
        self.ratings = {}            # (provider, model) -> rating row
        self.manifest = []           # workload requests, in order
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief.error = "the brief holds no provider directory"
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        pid = _string(raw.get("id"))
        wire = raw.get("wire")
        if not pid or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _string(raw.get(key)):
                base = _string(raw.get(key))
                break
        brief.providers.append({
            "id": pid,
            "name": _string(raw.get("name")) or pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": _string(raw.get("key_env")),
            "key": _string(raw.get("key")),
        })
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = _string(raw.get("model"))
            provider = _string(raw.get("provider"))
            if not model or not provider:
                continue
            brief.ratings[(provider, model)] = {
                "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) and not isinstance(raw.get("rating_milli"), bool) else 0,
                "context_length": _number_or_none(raw.get("context_length")),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            }
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw):
                continue
            rid = _string(raw.get("id"))
            if not rid:
                continue
            requires = raw.get("requires") if is_record(raw.get("requires")) else {}
            req = {
                "tools": requires.get("tools") is True,
                "vision": requires.get("vision") is True,
                "json": requires.get("json") is True,
            }
            if isinstance(requires.get("min_context"), (int, float)) and not isinstance(requires.get("min_context"), bool):
                req["min_context"] = requires["min_context"]
            brief.manifest.append({
                "id": rid,
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": req,
            })
    return brief


# --------------------------------------------------------------------------
# HTTP helpers (upstream calls to the env and its providers).
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.startswith("http://") or b.startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if not b:
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers=None, body=None, timeout=UPSTREAM_TIMEOUT_SEC):
    """One HTTP call; returns (status, bytes). Raises on transport errors."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


# --------------------------------------------------------------------------
# The catalog probe: each provider's model list, read in its own wire.
# --------------------------------------------------------------------------

class Probe:
    def __init__(self, provider):
        self.provider = provider
        self.ok = False
        self.models = []             # probed models, in the listing's order
        self.error = None


def _zero_priced(model):
    pricing = model.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for value in pricing.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, str) and value.strip():
            try:
                values.append(float(value))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def _model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        mid = _string(raw.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _string(raw.get("id"))
    if not mid:
        return None
    context = _number_or_none(raw.get("context_length"))
    if context is None:
        context = _number_or_none(raw.get("context_window"))
    if context is None:
        context = _number_or_none(raw.get("inputTokenLimit"))
    free = _bool_or_none(raw.get("free"))
    if free is None:
        free = _zero_priced(raw)
    display = _string(raw.get("display_name")) or _string(raw.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(raw.get("tools")),
        "vision": _bool_or_none(raw.get("vision")),
        "json": _bool_or_none(raw.get("json")),
    }


def probe_provider(env_url, provider):
    probe = Probe(provider)
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, raw = http_request("GET", url, headers=headers, timeout=PROBE_TIMEOUT_SEC)
    except Exception as err:  # transport failure: the provider sits this trial out
        probe.error = "GET %s failed: %s" % (url, err)
        return probe
    if status < 200 or status >= 300:
        probe.error = "GET %s answered HTTP %s" % (url, status)
        return probe
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        probe.error = "GET %s answered a non-JSON body" % url
        return probe
    if not is_record(body):
        probe.error = "GET %s answered a non-object body" % url
        return probe
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(listing, list):
        probe.error = "GET %s answered no model list" % url
        return probe
    for raw_model in listing:
        model = _model_in(provider["wire"], raw_model)
        if model is not None:
            probe.models.append(model)
    probe.ok = True
    return probe


def probe_all(env_url, providers):
    if not providers:
        return []
    with ThreadPoolExecutor(max_workers=min(10, len(providers))) as pool:
        return list(pool.map(lambda p: probe_provider(env_url, p), providers))


# --------------------------------------------------------------------------
# Model facts: the quality table is the sealed truth where it has a row;
# the provider's own listing speaks where it does not.
# --------------------------------------------------------------------------

def rating_row(brief, provider_id, model_id):
    return brief.ratings.get((provider_id, model_id))


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def model_caps(model, row):
    if row is not None:
        return {
            "context_length": row["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        if caps["context_length"] is None or caps["context_length"] < min_context:
            return False
    return True


def build_catalog(probes, brief):
    """The catalog document: `--list` prints it, /register posts it, one object.

    Models keep the order the provider's listing gave: the catalog is scored
    against the providers' true catalogs in that order. base_url is the
    directory's own path, never joined to the env's URL.
    """
    providers = []
    for probe in probes:
        if not probe.ok:
            continue
        seen = set()
        models = []
        for model in probe.models:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(brief, probe.provider["id"], model["id"])
            context = row["context_length"] if row is not None else model["context_length"]
            models.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": context,
            })
        providers.append({
            "id": probe.provider["id"],
            "base_url": probe.provider["base"],
            "key_env": probe.provider["key_env"],
            "wire": probe.provider["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, brief):
    """The free models this server answers for, as GET /v1/models lists them."""
    out = []
    for probe in probes:
        if not probe.ok:
            continue
        for model in sorted(probe.models, key=lambda m: m["id"]):
            row = rating_row(brief, probe.provider["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({
                "id": model["id"],
                "provider": probe.provider["id"],
                "display_name": model["display_name"],
            })
    return out


# --------------------------------------------------------------------------
# Canonical requests: a client body, in either client wire, as one shape.
#
#   system    the system prompt's whole text ("" when there is none)
#   messages  OpenAI-shaped: user/assistant/tool messages; a user or assistant
#             content is a string or a list of text / image_url parts, and an
#             assistant may carry tool_calls [{id, name, arguments(string)}]
#   tools     [{name, description, parameters}]
#   choice    "auto" | "required" | {"name": ...} | None
#   stops     [str], max_tokens, temperature, top_p
#   json_mode the client asked for a JSON answer (OpenAI response_format)
#   response_format  the client's own response_format object, when it sent one
# --------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = blocks_text(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def _arguments_object(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if is_record(parsed) else {}
    return {}


def canonical_from_openai(body):
    canon = {
        "system": "",
        "messages": [],
        "tools": [],
        "choice": None,
        "stops": [],
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "json_mode": False,
        "response_format": None,
    }
    system_parts = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = blocks_text(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            calls = []
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                calls.append({
                    "id": _string(call.get("id")),
                    "name": _string(fn.get("name")),
                    "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {}),
                })
            message = {"role": "assistant", "content": raw.get("content")}
            if calls:
                message["tool_calls"] = calls
            canon["messages"].append(message)
            continue
        if role == "tool":
            canon["messages"].append({
                "role": "tool",
                "tool_call_id": _string(raw.get("tool_call_id")),
                "content": raw.get("content") if isinstance(raw.get("content"), str) else _tool_result_text(raw.get("content")),
            })
            continue
        # user (and anything else) keeps its content untouched
        canon["messages"].append({"role": "user", "content": raw.get("content")})
    canon["system"] = "\n".join(system_parts)
    for raw in body.get("tools") or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        canon["tools"].append({
            "name": fn["name"],
            "description": _string(fn.get("description")),
            "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    choice = body.get("tool_choice")
    if choice == "auto":
        canon["choice"] = "auto"
    elif choice == "required":
        canon["choice"] = "required"
    elif is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        canon["choice"] = {"name": choice["function"]["name"]}
    max_tokens = _number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _number_or_none(body.get("max_completion_tokens"))
    canon["max_tokens"] = max_tokens
    canon["temperature"] = _number_or_none(body.get("temperature"))
    canon["top_p"] = _number_or_none(body.get("top_p"))
    stop = body.get("stop")
    if isinstance(stop, str):
        canon["stops"] = [stop]
    elif isinstance(stop, list):
        canon["stops"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format")
    if is_record(fmt):
        canon["response_format"] = fmt
        if fmt.get("type") in ("json_object", "json_schema"):
            canon["json_mode"] = True
    return canon


def canonical_from_anthropic(body):
    canon = {
        "system": blocks_text(body.get("system")),
        "messages": [],
        "tools": [],
        "choice": None,
        "stops": [],
        "max_tokens": _number_or_none(body.get("max_tokens")),
        "temperature": _number_or_none(body.get("temperature")),
        "top_p": _number_or_none(body.get("top_p")),
        "json_mode": False,
        "response_format": None,
    }
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            canon["messages"].append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                calls.append({
                    "id": _string(block.get("id")),
                    "name": _string(block.get("name")),
                    "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                })
            text = blocks_text(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                message["tool_calls"] = calls
            canon["messages"].append(message)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                canon["messages"].append({
                    "role": "tool",
                    "tool_call_id": _string(block.get("tool_use_id")),
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _anthropic_part_to_openai(block)
            if part is not None:
                parts.append(part)
        if parts:
            canon["messages"].append({"role": "user", "content": parts})
    for raw in body.get("tools") or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        canon["tools"].append({
            "name": raw["name"],
            "description": _string(raw.get("description")),
            "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            canon["choice"] = "auto"
        elif choice.get("type") == "any":
            canon["choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            canon["choice"] = {"name": choice["name"]}
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        canon["stops"] = [s for s in stops if isinstance(s, str)]
    return canon


def _anthropic_part_to_openai(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def requirements_from_body(body, wire):
    messages = body.get("messages") or []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if wire == "openai" else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body.get("tools")) > 0,
        "vision": vision,
        "json": is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"),
    }


# --------------------------------------------------------------------------
# Provider requests: the canonical request in each provider's own wire.
# The echo a provider computes (system text, user text, tool names,
# max_tokens, stops, image hash, JSON mode) must match the client's exactly,
# so every field is carried across, none rewritten or dropped.
# --------------------------------------------------------------------------

def build_openai_request(canon, model):
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    for message in canon["messages"]:
        if message["role"] == "assistant":
            out = {"role": "assistant", "content": message.get("content")}
            if message.get("tool_calls"):
                out["tool_calls"] = [{
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                } for call in message["tool_calls"]]
            messages.append(out)
        elif message["role"] == "tool":
            messages.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
        else:
            messages.append({"role": "user", "content": message.get("content")})
    body = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stops"]:
        body["stop"] = list(canon["stops"])
    if canon["tools"]:
        body["tools"] = [{
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
        } for t in canon["tools"]]
    if canon["choice"] == "auto":
        body["tool_choice"] = "auto"
    elif canon["choice"] == "required":
        body["tool_choice"] = "required"
    elif is_record(canon["choice"]):
        body["tool_choice"] = {"type": "function", "function": {"name": canon["choice"]["name"]}}
    if canon["response_format"] is not None:
        body["response_format"] = canon["response_format"]
    return body


def _openai_part_to_anthropic(part):
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    if part.get("type") == "image_url" and is_record(part.get("image_url")):
        url = part["image_url"].get("url")
        if isinstance(url, str):
            if url.startswith("data:") and ";base64," in url:
                media = url[len("data:"):url.index(";base64,")]
                data = url[url.index(";base64,") + len(";base64,"):]
                return {"type": "image", "source": {"type": "base64", "media_type": media or "image/png", "data": data}}
            return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def build_anthropic_request(canon, model):
    messages = []
    for message in canon["messages"]:
        if message["role"] == "assistant":
            content = message.get("content")
            blocks = []
            if isinstance(content, str) and content != "":
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if is_record(part) and part.get("type") == "text":
                        blocks.append({"type": "text", "text": part["text"]})
            for call in message.get("tool_calls") or []:
                blocks.append({
                    "type": "tool_use",
                    "id": call["id"] or "toolu_1",
                    "name": call["name"],
                    "input": _arguments_object(call["arguments"]),
                })
            messages.append({"role": "assistant", "content": blocks if blocks else ""})
        elif message["role"] == "tool":
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}],
            })
        else:
            content = message.get("content")
            if isinstance(content, list):
                blocks = []
                for part in content:
                    if is_record(part):
                        converted = _openai_part_to_anthropic(part)
                        if converted is not None:
                            blocks.append(converted)
                messages.append({"role": "user", "content": blocks})
            else:
                messages.append({"role": "user", "content": content if isinstance(content, str) else ""})
    body = {"model": model, "messages": messages}
    if canon["system"] != "":
        body["system"] = canon["system"]
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stops"]:
        body["stop_sequences"] = list(canon["stops"])
    if canon["tools"]:
        body["tools"] = [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        } for t in canon["tools"]]
    if canon["choice"] == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif canon["choice"] == "required":
        body["tool_choice"] = {"type": "any"}
    elif is_record(canon["choice"]):
        body["tool_choice"] = {"type": "tool", "name": canon["choice"]["name"]}
    return body


def build_gemini_request(canon, model):
    contents = []
    tool_names = {}
    for message in canon["messages"]:
        if message["role"] == "assistant":
            parts = []
            content = message.get("content")
            if isinstance(content, str) and content != "":
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if is_record(part) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"text": part["text"]})
            for call in message.get("tool_calls") or []:
                tool_names[call["id"]] = call["name"]
                parts.append({"functionCall": {"name": call["name"], "args": _arguments_object(call["arguments"])}})
            contents.append({"role": "model", "parts": parts if parts else [{"text": ""}]})
        elif message["role"] == "tool":
            part = {"functionResponse": {
                "name": tool_names.get(message["tool_call_id"], ""),
                "response": {"result": message["content"]},
            }}
            # A tool result rides on the last user content: the provider reads
            # the last non-model content as the user turn, and a bare result
            # content would shadow the user's text there.
            for content in reversed(contents):
                if content["role"] == "user":
                    content["parts"].append(part)
                    break
            else:
                contents.append({"role": "user", "parts": [part]})
        else:
            parts = []
            content = message.get("content")
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not is_record(part):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                        url = part["image_url"].get("url")
                        if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                            media = url[len("data:"):url.index(";base64,")]
                            data = url[url.index(";base64,") + len(";base64,"):]
                            parts.append({"inlineData": {"mimeType": media or "image/png", "data": data}})
                        elif isinstance(url, str):
                            parts.append({"fileData": {"fileUri": url}})
            contents.append({"role": "user", "parts": parts if parts else [{"text": ""}]})
    body = {"contents": contents}
    if canon["system"] != "":
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        } for t in canon["tools"]]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = list(canon["stops"])
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        config["topP"] = canon["top_p"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Provider answers: one normalized shape {text, input_tokens, output_tokens}.
# --------------------------------------------------------------------------

def _text_blocks_joined(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def parse_provider_answer(wire, raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not is_record(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        message = choices[0].get("message")
        if not is_record(message):
            return None
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return {
            "text": _text_blocks_joined(message.get("content")),
            "input_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), (int, float)) else 0,
            "output_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), (int, float)) else 0,
        }
    if wire == "anthropic":
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return {
            "text": _text_blocks_joined(body.get("content")),
            "input_tokens": usage.get("input_tokens") if isinstance(usage.get("input_tokens"), (int, float)) else 0,
            "output_tokens": usage.get("output_tokens") if isinstance(usage.get("output_tokens"), (int, float)) else 0,
        }
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p["text"] for p in parts if is_record(p) and isinstance(p.get("text"), str))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {
        "text": text,
        "input_tokens": usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), (int, float)) else 0,
        "output_tokens": usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), (int, float)) else 0,
    }


def call_provider(env_url, provider, model_id, canon):
    """One inference call on the provider's own wire. Returns (status, answer)."""
    base = join_base(env_url, provider["base"])
    headers = dict(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "openai":
        url = base + "/v1/chat/completions"
        payload = build_openai_request(canon, model_id)
    elif provider["wire"] == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        payload = build_anthropic_request(canon, model_id)
    else:
        url = base + "/v1beta/models/" + model_id + ":generateContent"
        payload = build_gemini_request(canon, model_id)
    status, raw = http_request("POST", url, headers=headers, body=payload)
    if status < 200 or status >= 300:
        return status, None
    answer = parse_provider_answer(provider["wire"], raw)
    return (status, answer) if answer is not None else (502, None)


# --------------------------------------------------------------------------
# Client answers: the provider's answer in the client's wire, JSON or SSE.
# The provider's stop is always the natural-stop family, reported as "stop"
# on the OpenAI wire and "end_turn" on the Anthropic one; the provider's
# usage counts are reported as that wire spells them.
# --------------------------------------------------------------------------

def _sse(event, data):
    return "%sdata: %s\n\n" % (("event: %s\n" % event) if event else "", json.dumps(data))


def _split_text(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def render_openai(answer, model_id, stream):
    usage = {
        "prompt_tokens": answer["input_tokens"],
        "completion_tokens": answer["output_tokens"],
        "total_tokens": answer["input_tokens"] + answer["output_tokens"],
    }
    if not stream:
        return "application/json", json.dumps({
            "id": "chatcmpl-aggregate",
            "object": "chat.completion",
            "created": FIXED_OPENAI_CREATED,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer["text"]},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": usage,
        }).encode("utf-8")
    frames = []
    chunk = lambda delta, finish: {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion.chunk",
        "created": FIXED_OPENAI_CREATED,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    frames.append(_sse(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(answer["text"]):
        frames.append(_sse(None, chunk({"content": piece}, None)))
    frames.append(_sse(None, chunk({}, "stop")))
    frames.append(_sse(None, {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion.chunk",
        "created": FIXED_OPENAI_CREATED,
        "model": model_id,
        "choices": [],
        "usage": usage,
    }))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(frames).encode("utf-8")


def render_anthropic(answer, model_id, stream):
    if not stream:
        return "application/json", json.dumps({
            "id": "msg_aggregate",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [{"type": "text", "text": answer["text"]}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"]},
        }).encode("utf-8")
    frames = []
    frames.append(_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_aggregate",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": answer["input_tokens"], "output_tokens": 0},
        },
    }))
    frames.append(_sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(answer["text"]):
        frames.append(_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": answer["output_tokens"]},
    }))
    frames.append(_sse("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames).encode("utf-8")


def openai_error(message, err_type):
    return {"error": {"message": message, "type": err_type, "code": None}}


def anthropic_error(err_type, message):
    return {"type": "error", "error": {"type": err_type, "message": message}}


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{
            "id": m["id"],
            "object": "model",
            "created": FIXED_OPENAI_CREATED,
            "owned_by": m["provider"],
        } for m in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{
            "type": "model",
            "id": m["id"],
            "display_name": m["display_name"],
            "created_at": FIXED_ANTHROPIC_CREATED_AT,
        } for m in models],
        "has_more": False,
    }


# --------------------------------------------------------------------------
# Routing: every free candidate that meets the request, best rating first;
# a refusal moves the request to the next candidate. Attempts so far break
# rating ties, so equal models spread over their providers.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, probes):
        self.brief = brief
        self.probes = probes
        self.attempts = {}           # provider id -> inference calls made
        self.manifest_index = 0

    def candidates(self, requires, json_mode):
        found = []
        for order, probe in enumerate(self.probes):
            if not probe.ok:
                continue
            provider = probe.provider
            # A JSON-mode request must ride a wire that can carry it: the
            # Anthropic wire echoes JSON mode as absent, so it is skipped.
            if json_mode and provider["wire"] == "anthropic":
                continue
            for model in probe.models:
                row = rating_row(self.brief, provider["id"], model["id"])
                if not model_free(model, row):
                    continue
                if not meets_requirements(requires, model_caps(model, row)):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                found.append({
                    "rating": rating,
                    "attempts": self.attempts.get(provider["id"], 0),
                    "order": order,
                    "model_id": model["id"],
                    "probe": probe,
                })
        found.sort(key=lambda c: (-c["rating"], c["attempts"], c["order"], c["model_id"]))
        return found

    def serve(self, canon, requires):
        """Answer one canonical request; returns (provider, model_id, answer) or None."""
        for candidate in self.candidates(requires, canon["json_mode"]):
            provider = candidate["probe"].provider
            self.attempts[provider["id"]] = self.attempts.get(provider["id"], 0) + 1
            try:
                status, answer = call_provider(ENV_URL, provider, candidate["model_id"], canon)
            except Exception as err:
                log("provider %s failed: %s" % (provider["id"], err))
                continue
            if answer is not None:
                return provider, candidate["model_id"], answer
            log("provider %s answered HTTP %s; failing over" % (provider["id"], status))
        return None


# --------------------------------------------------------------------------
# The server: the two client routes, the model list, register and the poll.
# --------------------------------------------------------------------------

class State:
    def __init__(self):
        self.brief = Brief()
        self.probes = []
        self.catalog = {"providers": []}
        self.catalog_text = ""
        self.aggregator = None
        self.lock = threading.Lock()


STATE = State()


def load():
    STATE.brief = read_brief(PROMPT_FILE)
    if STATE.brief.error:
        log(STATE.brief.error)
    STATE.probes = probe_all(ENV_URL, STATE.brief.providers)
    for probe in STATE.probes:
        if not probe.ok and probe.error:
            log(probe.error)
    STATE.catalog = build_catalog(STATE.probes, STATE.brief)
    STATE.catalog_text = json.dumps(STATE.catalog, separators=(",", ":"))
    STATE.aggregator = Aggregator(STATE.brief, STATE.probes)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status, value):
        self._send(status, "application/json", json.dumps(value).encode("utf-8"))

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            models = served_models(STATE.probes, STATE.brief)
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_json(404, openai_error("no route GET %s" % self.path, "not_found"))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._handle_chat("openai")
        elif path == "/v1/messages":
            self._handle_chat("anthropic")
        else:
            self._send_json(404, openai_error("no route POST %s" % path, "not_found"))

    def _handle_chat(self, wire):
        length = self.headers.get("content-length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            size = 0
        if size > MAX_BODY_BYTES:
            self._fail(wire, 413, "the request body is too large")
            return
        raw = self.rfile.read(size) if size > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        if not is_record(body):
            self._fail(wire, 400, "the request body is not one JSON object")
            return
        with STATE.lock:
            self._answer(wire, body)

    def _fail(self, wire, status, message):
        if wire == "anthropic":
            self._send_json(status, anthropic_error("invalid_request_error", message))
        else:
            self._send_json(status, openai_error(message, "invalid_request_error"))

    def _answer(self, wire, body):
        agg = STATE.aggregator
        entry = None
        if agg.manifest_index < len(STATE.brief.manifest):
            entry = STATE.brief.manifest[agg.manifest_index]
        agg.manifest_index += 1
        requires = entry["requires"] if entry is not None else requirements_from_body(body, wire)
        canon = canonical_from_openai(body) if wire == "openai" else canonical_from_anthropic(body)
        stream = body.get("stream") is True
        served = agg.serve(canon, requires)
        if served is None:
            if wire == "anthropic":
                self._send_json(503, anthropic_error("overloaded_error", "no free model on the directory could serve this request"))
            else:
                self._send_json(503, openai_error("no free model on the directory could serve this request", "no_free_model"))
            return
        _provider, model_id, answer = served
        if wire == "openai":
            content_type, payload = render_openai(answer, model_id, stream)
        else:
            content_type, payload = render_anthropic(answer, model_id, stream)
        self._send(200, content_type, payload)


def register(port):
    # The register body's catalog is the --list document, byte for byte: the
    # body is assembled around the one serialized catalog text.
    payload = ('{"protocol_version": %d, "port": %d, "catalog": %s}'
               % (PROTOCOL_VERSION, port, STATE.catalog_text)).encode("utf-8")
    req = urllib.request.Request(
        ENV_URL + "/register", data=payload,
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SEC) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    if status < 200 or status >= 300:
        raise RuntimeError("POST /register answered HTTP %s: %s" % (status, raw[:200]))
    reply = json.loads(raw.decode("utf-8"))
    if is_record(reply) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %s requests" % reply["requests"])


def poll_work():
    while True:
        try:
            status, raw = http_request("GET", ENV_URL + "/work", timeout=2.0)
            if status == 200:
                work = json.loads(raw.decode("utf-8"))
                if is_record(work) and work.get("done") is True:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)
        except Exception:
            pass
        threading.Event().wait(POLL_INTERVAL_SEC)


def main():
    load()
    if "--list" in sys.argv:
        sys.stdout.write(STATE.catalog_text + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        server.server_close()
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = server.server_address[1]
    log("listening on 0.0.0.0:%s with %s answered providers and %s free models" % (
        port,
        len(STATE.catalog["providers"]),
        len(served_models(STATE.probes, STATE.brief)),
    ))
    register(port)
    thread = threading.Thread(target=poll_work, daemon=True)
    thread.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
