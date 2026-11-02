# ADR 0002: Run inference inside the request

**Status:** accepted

**Context:** A queue and workers add moving parts, and one image takes well under a second on CPU.

**Decision:** Run inference in-process during the request.

**Consequences:** Simple to run and debug, but one process handles one inference at a time. For heavier traffic, move inference to a thread pool or a queue.
