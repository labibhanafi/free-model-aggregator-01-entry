# Free Model Aggregator

An entry for make.fun's **free-model-aggregator-01** bounty: one server, two
client APIs, ten simulated free-tier providers.

`main.py` (Python 3.13, standard library only) listens on a port it picks,
registers with the trial env, and answers the env client's 20 requests:

- **Catalog** — every provider is probed in its own wire (OpenAI, Anthropic,
  Gemini); the catalog document that `--list` prints is the one registered,
  models in each listing's own order with the quality table's free flags.
- **Routing** — for each request, every free model that meets its requirements
  (tools, vision, JSON mode, minimum context) is ranked by the quality table's
  `rating_milli`; rating ties spread over providers by attempts so far.
- **Wires** — requests are translated between the client's wire and the
  provider's wire field for field: system prompt, user text, images, tools,
  stop sequences, `max_tokens`, JSON mode. JSON-mode requests never ride the
  Anthropic wire, which cannot carry JSON mode. Answers come back in the
  client's wire, JSON or server-sent events, with the provider's text whole,
  its usage counts and its stop reason mapped (`end_turn` ↔ `stop`).
- **Failover** — a refused call (402, 429, 5xx) is counted and the request
  moves to the next-best candidate until one serves it; a paid model is never
  a legal pick.

Everything is a pure function of the brief, the requests and the providers'
answers — no wall clock, no randomness — so a replayed trial reproduces the
same state export byte for byte.

Run it the way a trial does:

```
MAKEFUN_ENV_URL=http://store:8080 MAKEFUN_PROMPT_FILE=/input/prompt.txt python3 main.py
python3 main.py --list   # prints the catalog document and exits
```

Licensed under the MIT License (see `LICENSE`).
