# Cached Agent Workflow

The cached agent workflow reduces LLM cost and latency by using a three-layer cache.

Layer 1 is exact cache. It returns a previous answer when the same request appears again.
Layer 2 is semantic cache. It retrieves similar previous requests using embeddings.
Layer 3 is tool/result cache. It reuses expensive tool outputs when the underlying input has not changed.

Observability includes request traces, cache hit ratio, latency, token usage, and error rates.
