# Python Structured Specialist — Final Architecture & Dataset Design

**Status:** Design specification before dataset generation  
**Primary base model:** `Salesforce/codet5p-220m-py`  
**Model family:** CodeT5+ encoder-decoder, Python-specialized  
**Primary language:** Python  
**Primary goal:** Fine-tune a compact Python specialist that can generate, repair, refactor, and produce complete Python files while making strong software-engineering decisions and emitting a constrained Python-oriented intermediate representation (IR).

---

# Table of Contents

1. [Project Goal](#1-project-goal)
2. [What the Model Is Expected to Know](#2-what-the-model-is-expected-to-know)
3. [Four Core Capabilities](#3-four-core-capabilities)
4. [Primary Base Model](#4-primary-base-model)
5. [Core Design Principle](#5-core-design-principle)
6. [Final Architecture](#6-final-architecture)
7. [Responsibility of Each Architecture Stage](#7-responsibility-of-each-architecture-stage)
8. [Model Responsibility vs Deterministic Responsibility](#8-model-responsibility-vs-deterministic-responsibility)
9. [Python-Oriented IR](#9-python-oriented-ir)
10. [Why the Decoder Should Not Emit JSON or Raw AST](#10-why-the-decoder-should-not-emit-json-or-raw-ast)
11. [Tokenizer and Special Tokens](#11-tokenizer-and-special-tokens)
12. [Handling Existing and New Libraries](#12-handling-existing-and-new-libraries)
13. [Library/API Context Mechanism](#13-libraryapi-context-mechanism)
14. [Detailed Task Behavior](#14-detailed-task-behavior)
15. [Final Dataset Schema](#15-final-dataset-schema)
16. [Dataset Field Definitions](#16-dataset-field-definitions)
17. [Dataset Coverage Requirements](#17-dataset-coverage-requirements)
18. [Difficulty Distribution](#18-difficulty-distribution)
19. [Prompt Style Distribution](#19-prompt-style-distribution)
20. [Domain Coverage](#20-domain-coverage)
21. [New-Library Generalization Data](#21-new-library-generalization-data)
22. [Dataset Examples](#22-dataset-examples)
23. [IR Examples](#23-ir-examples)
24. [Dataset Generation Strategy](#24-dataset-generation-strategy)
25. [Teacher-Model Strategy](#25-teacher-model-strategy)
26. [Deterministic IR Construction](#26-deterministic-ir-construction)
27. [Verification Rules](#27-verification-rules)
28. [Deduplication and Contamination Control](#28-deduplication-and-contamination-control)
29. [Training Input and Target Format](#29-training-input-and-target-format)
30. [Inference Pipeline](#30-inference-pipeline)
31. [Evaluation Plan](#31-evaluation-plan)
32. [Ablation Plan](#32-ablation-plan)
33. [Scaling Plan](#33-scaling-plan)
34. [Open Research Questions](#34-open-research-questions)
35. [Final Design Summary](#35-final-design-summary)

---

# 1. Project Goal

The project is to build a compact but strong **Python specialist model**.

The model is not a generic conversational assistant. Its job is to transform a programming request, codebase context, existing Python code, or an error into a structured implementation representation that can be deterministically compiled into Python.

The central research hypothesis is:

> A relatively small encoder-decoder model may perform better on Python software-engineering tasks if it learns to produce a structured Python-oriented IR instead of spending capacity on indentation, punctuation, formatting, and unconstrained source-code syntax.

The model should generalize across Python use cases rather than being narrowly specialized to one framework or domain.

```text
user requirement / code / traceback / project context
                         ↓
                 CodeT5+ encoder
                         ↓
              program understanding
                         ↓
                structured decoder
                         ↓
              Python-oriented IR
                         ↓
             deterministic compiler
                         ↓
                    AST/CST
                         ↓
                  Python source
                         ↓
                  verification
```

The deterministic layer handles mechanical syntax construction. The neural model handles decisions that require software-engineering knowledge or judgment.

---

# 2. What the Model Is Expected to Know

The model is not learning Python from zero.

Because the project fine-tunes a pretrained coding model, it begins with knowledge acquired during pretraining. Fine-tuning should reinforce, specialize, and reorganize that knowledge toward our desired behavior.

The model may already contain useful knowledge about:

- Python syntax and semantics
- Python standard library
- algorithms and data structures
- object-oriented and functional programming
- web frameworks
- database libraries
- testing libraries
- data-science libraries
- machine-learning libraries
- networking
- filesystem and OS interaction
- CLI development
- package/module organization
- common programming patterns

The specialist dataset should strongly reinforce:

- idiomatic Python
- maintainability and readability
- typing
- API design
- database design
- security
- authentication
- authorization
- concurrency and asyncio
- transactions
- resource management
- error handling
- testing
- performance
- packaging and architecture
- appropriate SOLID use
- appropriate design-pattern use
- avoiding unnecessary abstraction
- respecting existing project conventions
- integrating unfamiliar libraries from supplied context
- repairing real bugs
- behavior-preserving refactoring

The model should learn **engineering judgment**, not ritualized pattern usage.

Example:

```text
simple problem
→ simple implementation
```

not:

```text
simple problem
→ Factory + Strategy + Repository + Protocol + dependency container
```

---

# 3. Four Core Capabilities

## 3.1 Generate

Input:

```text
Create an async TTL cache.
```

The model should:

- understand the requirement
- make sensible defaults when details are omitted
- choose appropriate Python constructs and APIs
- select libraries when needed
- emit build IR
- allow deterministic compilation to Python

Typical scope:

- function
- class
- utility
- algorithm
- service component
- data transformation
- client
- repository
- helper

## 3.2 Full-File Generate

Input:

```text
Create a complete Python module implementing a rate-limited async HTTP client.
```

The result must represent a complete `.py` module, including as appropriate:

- imports
- constants/configuration
- data models
- classes
- functions
- errors
- resource lifecycle
- typing
- useful docstrings/comments
- public API

No omitted sections or placeholders.

## 3.3 Repair

Input may be vague:

```text
Fix this.
```

plus code, or more explicit:

```text
This occasionally deadlocks under concurrent load.
```

or contain a traceback/test failure.

The model must infer the problem from:

- code
- errors
- tracebacks
- tests
- project context
- library usage

The repair should be minimal when possible and architectural when necessary.

## 3.4 Refactor

Input may be:

```text
Refactor this.
```

The model must inspect working code and improve it appropriately.

Possible improvements:

- separation of concerns
- reduced duplication
- stronger typing
- resource lifecycle
- testability
- simpler control flow
- dependency injection
- better abstraction boundaries
- appropriate design patterns
- removal of unnecessary design patterns
- security
- performance
- async correctness
- database transaction boundaries

Behavior should be preserved unless the user explicitly requests a behavior change.

---

# 4. Primary Base Model

The primary student model is:

```text
Salesforce/codet5p-220m-py
```

Why it is a good first fit:

- encoder-decoder architecture
- Python-specialized checkpoint
- manageable size for limited compute
- already contains substantial coding knowledge
- natural fit for input → structured-output fine-tuning
- suitable for comparing direct code generation against IR generation

The project should not assume 220M is the final optimal size.

The broader research question remains:

> How small can a Python specialist become before software-engineering capability collapses?

---

# 5. Core Design Principle

The most important architecture rule is:

> **Anything requiring knowledge, judgment, design choice, API knowledge, library selection, or semantic understanding belongs to the model.**

> **Anything that is purely mechanical syntax construction belongs to deterministic code.**

This prevents the deterministic layer from becoming a hard-coded knowledge base for FastAPI, Flask, Django, SQLAlchemy, Pandas, PyTorch, or any other library.

The deterministic compiler understands **Python constructs**, not third-party library semantics.

---

# 6. Final Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│                                                             │
│ requirement                                                 │
│ existing Python                                             │
│ traceback/error                                             │
│ tests                                                       │
│ project context                                             │
│ dependency context                                          │
│ library docs/signatures/examples                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT SERIALIZER / CONTEXT BUILDER                       │
│                                                             │
│ deterministic                                               │
│                                                             │
│ combines available input into a stable tagged format        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CODET5+ ENCODER                                         │
│                                                             │
│ neural                                                      │
│                                                             │
│ understands requirement, code, errors, project context,     │
│ library APIs, constraints and relationships                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. INTERNAL PROGRAM REPRESENTATION                          │
│                                                             │
│ neural hidden state                                         │
│                                                             │
│ intent                                                      │
│ symbols                                                     │
│ scope                                                       │
│ code structure                                              │
│ data relationships                                          │
│ type hints                                                  │
│ error context                                               │
│ selected constraints                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. STRUCTURED DECODER                                       │
│                                                             │
│ neural                                                      │
│                                                             │
│ decides WHAT implementation should exist                    │
│                                                             │
│ architecture                                                │
│ algorithms                                                  │
│ library/framework usage                                     │
│ API behavior                                                │
│ DB behavior                                                 │
│ auth/authz                                                  │
│ security                                                    │
│ transactions                                                │
│ concurrency                                                 │
│ types                                                       │
│ error handling                                              │
│ control flow                                                │
│ calls                                                       │
│ imports                                                     │
│ repair/refactor changes                                     │
│                                                             │
│ output: Python-oriented IR                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. IR GRAMMAR CONSTRAINT ENGINE                             │
│                                                             │
│ deterministic                                               │
│                                                             │
│ masks illegal structural tokens during decoding             │
│ validates structural nesting                                │
│ guarantees IR grammar validity                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. IR → PYTHON AST/CST COMPILER                             │
│                                                             │
│ deterministic                                               │
│                                                             │
│ IMPORT           → import node                              │
│ FUNCTION         → FunctionDef                              │
│ ASYNC_FUNCTION   → AsyncFunctionDef                         │
│ CALL             → Call                                     │
│ ATTRIBUTE        → Attribute                                │
│ AWAIT            → Await                                    │
│ IF               → If                                       │
│ RETURN           → Return                                   │
│ CLASS            → ClassDef                                 │
│ DECORATOR        → decorator list                           │
│ etc.                                                        │
│                                                             │
│ NO library-specific engineering decisions happen here       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. CST / SOURCE RENDERER                                    │
│                                                             │
│ deterministic                                               │
│                                                             │
│ comments                                                    │
│ docstrings                                                  │
│ spacing                                                     │
│ indentation                                                 │
│ source formatting                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    PYTHON SOURCE
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. VERIFICATION                                             │
│                                                             │
│ deterministic                                               │
│                                                             │
│ ast.parse                                                   │
│ compile                                                     │
│ test execution                                              │
│ optional type checker                                       │
│ optional lint/static checks                                 │
└────────────────────┬───────────────────────┬────────────────┘
                     │                       │
                   PASS                    FAIL
                     │                       │
                     ▼                       ▼
                 FINAL CODE          verification feedback
                                             │
                                             ▼
                                     model repair iteration
```

---

# 7. Responsibility of Each Architecture Stage

## 7.1 Input Serializer / Context Builder

Deterministic. It only organizes available information.

Example:

```text
<TASK>
repair
</TASK>

<INSTRUCTION>
Fix this.
</INSTRUCTION>

<PROJECT_CONTEXT>
Python 3.12
asyncio application
</PROJECT_CONTEXT>

<LIBRARY_CONTEXT>
httpx.AsyncClient.get(...)
httpx.Response.raise_for_status()
</LIBRARY_CONTEXT>

<CODE_BEFORE>
async def fetch(client, url):
    return client.get(url)
</CODE_BEFORE>

<ERROR>
RuntimeWarning: coroutine was never awaited
</ERROR>
```

## 7.2 CodeT5+ Encoder

Neural. It understands relationships among:

- requirement text
- code
- names
- types
- errors
- APIs
- project conventions
- library context
- tests
- user constraints

It must distinguish explicit requirements from defaults.

Example:

```text
User: Use Flask.
```

means Flask is a hard constraint.

```text
User: Create a REST API.
```

allows the model to choose an appropriate framework.

## 7.3 Internal Program Representation

This is hidden neural state, not a required textual reasoning output.

It should encode:

- desired behavior
- existing program organization
- symbols and scope
- dependencies
- code structure
- probable bug source
- relevant APIs
- required transformations

## 7.4 Structured Decoder

This is the main intelligent component.

It decides:

- what to build
- what to change
- how to structure it
- which library/framework API to use
- whether to use async
- resource cleanup
- error modeling
- transaction boundaries
- locking/concurrency
- design patterns when appropriate
- data structures
- classes/functions
- imports and calls
- parameters and types
- expressions and control flow

It emits **Python-oriented IR**.

## 7.5 IR Grammar Constraint Engine

Deterministic. It validates only IR grammar.

Examples:

- `<RETURN>` outside a function can be blocked.
- `<AWAIT>` in an invalid synchronous context can be blocked.
- `<END_FUNCTION>` without an open function can be blocked.
- `<ELSE>` without a matching `<IF>` can be blocked.

This lets the decoder spend less capacity learning delimiter syntax.

## 7.6 IR → AST/CST Compiler

Deterministic and intentionally library-agnostic.

It understands generic constructs such as:

```text
IMPORT
IMPORT_FROM
ASSIGN
CALL
ATTRIBUTE
FUNCTION
ASYNC_FUNCTION
CLASS
DECORATOR
IF
FOR
WHILE
TRY
EXCEPT
WITH
ASYNC_WITH
AWAIT
RETURN
RAISE
COMPREHENSION
LAMBDA
MATCH
```

Third-party APIs are represented as normal names, attributes, calls, arguments, and decorators.

## 7.7 CST / Source Renderer

Produces readable source and handles:

- indentation
- spaces
- line breaks
- comments
- docstrings
- formatting conventions

## 7.8 Verification

Deterministic checks may include:

```text
ast.parse()
compile()
pytest
type checker
ruff/static analysis
custom behavioral checks
```

Verification should be task-aware.

---

# 8. Model Responsibility vs Deterministic Responsibility

| Decision / Operation | Model | Deterministic |
|---|---:|---:|
| Understand requirement | ✓ | |
| Understand existing code | ✓ | |
| Infer bug | ✓ | |
| Select FastAPI / Flask / Django | ✓ | |
| Respect user-selected framework | ✓ | |
| Continue existing project framework | ✓ | |
| Choose SQLAlchemy / raw SQL / other DB layer | ✓ | |
| Decide transaction boundary | ✓ | |
| Decide authorization behavior | ✓ | |
| Decide security checks | ✓ | |
| Choose Strategy / Factory / Repository | ✓ | |
| Decide not to use a pattern | ✓ | |
| Select asyncio/threading/multiprocessing | ✓ | |
| Use supplied new-library API | ✓ | |
| Decide imports and API calls | ✓ | |
| Decide class/function structure | ✓ | |
| Construct `ast.FunctionDef` | | ✓ |
| Construct `ast.Call` | | ✓ |
| Construct `ast.Await` | | ✓ |
| Indentation | | ✓ |
| IR grammar validation | | ✓ |
| Python parsing | | ✓ |
| Compilation | | ✓ |
| Test execution | | ✓ |

---

# 9. Python-Oriented IR

The decoder should not emit a giant CPython AST dump.

It should emit a compact structured language that is:

- expressive enough for arbitrary Python
- stable
- constrainable
- easy to compile
- reasonably compact
- library-agnostic
- Python-aware

Example:

```text
<BUILD>

<IMPORT_FROM>
module fastapi
names APIRouter, Depends
<END_OP>

<ASSIGN>
target router
value <CALL> APIRouter <END_CALL>
<END_OP>

<ASYNC_FUNCTION>
name create_user

<PARAM>
name request
type CreateUserRequest
<END_PARAM>

<DECORATOR>
<CALL>
function router.post
arg "/users"
<END_CALL>
<END_DECORATOR>

<RETURN>
<CALL>
function service.create_user
arg request
<END_CALL>
<END_RETURN>

<END_FUNCTION>

<END>
```

The compiler does not know what `APIRouter` means. The model does.

---

# 10. Why the Decoder Should Not Emit JSON or Raw AST

Avoid training targets such as:

```json
{
  "operation": "EDIT_PROGRAM",
  "added_node_types": {
    "Name": 10,
    "Call": 5,
    "Load": 10
  }
}
```

Problems:

1. Node counts are not semantically useful.
2. JSON punctuation wastes decoder capacity.
3. JSON can become malformed.
4. Raw AST is excessively verbose.
5. Library/API intent becomes difficult to represent cleanly.
6. A small model must learn too much serialization syntax.

Also avoid unconstrained natural-language pseudocode because it cannot be deterministically compiled reliably.

The target should be a formal IR with a known grammar.

---

# 11. Tokenizer and Special Tokens

Extend the CodeT5+ tokenizer with a moderate number of stable structural tokens.

Example candidates:

```text
<BUILD>
<EDIT>
<END>

<IMPORT>
<IMPORT_FROM>
<ASSIGN>
<FUNCTION>
<ASYNC_FUNCTION>
<CLASS>
<PARAM>
<RETURN>
<CALL>
<ATTRIBUTE>
<AWAIT>
<IF>
<ELSE>
<FOR>
<WHILE>
<TRY>
<EXCEPT>
<FINALLY>
<WITH>
<ASYNC_WITH>
<RAISE>
<DECORATOR>
<YIELD>
<MATCH>

<INSERT>
<DELETE>
<REPLACE>
<MOVE>
<RENAME>

<NAME>
<TYPE>
<TARGET>
<VALUE>
<FROM>
<TO>

<END_OP>
<END_CALL>
<END_FUNCTION>
<END_CLASS>
```

Do not create special tokens for every variable, package, framework, class, API, or design pattern.

Names such as:

```text
fastapi
sqlalchemy
pandas
hyperdb
UserRepository
fetch_user
```

remain normal text/subword tokens.

After adding special tokens, resize the model's token embeddings before fine-tuning.

---

# 12. Handling Existing and New Libraries

There are three main cases.

## 12.1 Library Known From Pretraining

Example:

```text
Use SQLAlchemy 2 async sessions.
```

The base model may already know common APIs. Fine-tuning teaches it to express that implementation through our IR.

## 12.2 Existing Project Library

Existing source:

```python
from flask import Blueprint
```

User:

```text
Add a /users endpoint.
```

The model should recognize the existing Flask project and continue that architecture unless instructed otherwise.

Priority:

```text
explicit user requirement
    ↓
existing project conventions
    ↓
existing dependencies/imports
    ↓
compatibility
    ↓
learned sensible default
```

## 12.3 Completely New or Unfamiliar Library

The model cannot know an unseen API magically.

Provide **library context**.

Example:

```text
<LIBRARY_CONTEXT>

Package: acmecache
Version: 1.4

API:
CacheClient(url: str)

await CacheClient.get(key: str) -> bytes | None
await CacheClient.set(
    key: str,
    value: bytes,
    ttl: int
) -> None

Usage:
client = CacheClient("redis://localhost")
value = await client.get("user:1")

</LIBRARY_CONTEXT>
```

Requirement:

```text
Create an async user repository that caches loaded users for 60 seconds.
```

The model learns to integrate the supplied API. The deterministic compiler remains unchanged.

---

# 13. Library/API Context Mechanism

Library context may come from:

- user-provided documentation
- existing project source
- installed package inspection
- `.pyi` type stubs
- function signatures
- class signatures
- module exports
- docstrings
- relevant usage examples
- retrieved documentation

Possible deterministic inspector:

```text
installed package
      ↓
inspect signatures
      ↓
read type stubs
      ↓
extract public classes/functions
      ↓
compress relevant API surface
      ↓
<LIBRARY_CONTEXT>
```

The model performs the intelligent integration.

The deterministic compiler still has no special knowledge of the library.

---

# 14. Detailed Task Behavior

## Generate

Example:

```text
Build a retry helper for async network calls.
```

The model may decide:

- exponential backoff
- bounded retry count
- cancellation should propagate
- retry only selected exceptions
- preserve final exception
- add jitter when appropriate

## Full-File Generate

Example:

```text
Create a complete module implementing an async HTTP client with bounded concurrency and retries.
```

The IR may include:

- imports
- configuration
- exceptions
- client class
- lifecycle
- semaphore
- retry logic
- typing
- return behavior

## Repair

Input:

```text
Fix this.
```

```python
async def fetch(url, client):
    return client.get(url)
```

Error:

```text
RuntimeWarning: coroutine was never awaited
```

Expected semantic change:

```text
await client.get(url)
```

## Refactor

Input:

```text
Refactor this.
```

The model should independently identify whether useful improvements involve extraction, simplification, dependency injection, typing, reduced coupling, better async/resource management, security, or performance.

---

# 15. Final Dataset Schema

Recommended final dataset format:

```text
id
task
instruction
code_before
error
project_context
library_context
tests
target_mode
target_ir
target_code
domain
subdomain
difficulty
instruction_style
concepts
libraries
verified
verification_level
source_type
source_dataset
source_reference
ir_version
content_hash
```

**24 columns**.

During generation, JSONL should be the working format. CSV can be exported for inspection, but multiline source and nested metadata are easier to manage in JSONL.

---

# 16. Dataset Field Definitions

| Field | Purpose |
|---|---|
| `id` | Unique stable row identifier |
| `task` | `generate`, `full_file_generate`, `repair`, `refactor` |
| `instruction` | Realistic user request |
| `code_before` | Existing/broken code for repair/refactor |
| `error` | Optional traceback, failing test, warning, or behavior description |
| `project_context` | Existing project conventions, symbols, architecture, versions |
| `library_context` | Relevant API docs, signatures, examples, type-stub excerpts |
| `tests` | Verification code or behavioral checks |
| `target_mode` | `build` or `edit` |
| `target_ir` | Final formal IR sequence produced by the decoder |
| `target_code` | Ground-truth Python source |
| `domain` | Broad category |
| `subdomain` | More specific category |
| `difficulty` | beginner/intermediate/advanced/expert |
| `instruction_style` | vague/normal/detailed/expert_constrained |
| `concepts` | JSON list of concepts covered |
| `libraries` | JSON list of libraries/APIs involved |
| `verified` | Whether required verification was completed |
| `verification_level` | structural/tests/behavioral/static/full |
| `source_type` | teacher_synthetic/public_transformed/mutated/manual/etc. |
| `source_dataset` | Source dataset/model name |
| `source_reference` | Traceable source identifier when applicable |
| `ir_version` | IR schema version, e.g. `v1` |
| `content_hash` | Stable deduplication hash |

---

# 17. Dataset Coverage Requirements

Mature target:

```text
200,000 accepted examples
```

Suggested exact task balance:

| Task | Rows |
|---|---:|
| Generate | 50,000 |
| Full-file Generate | 50,000 |
| Repair | 50,000 |
| Refactor | 50,000 |
| **Total** | **200,000** |

Do not jump directly to 200K. Validate progressively.

---

# 18. Difficulty Distribution

Within each 50K task:

| Difficulty | Percentage | Rows per task |
|---|---:|---:|
| Beginner | 15% | 7,500 |
| Intermediate | 35% | 17,500 |
| Advanced | 35% | 17,500 |
| Expert | 15% | 7,500 |

Whole 200K dataset:

| Difficulty | Rows |
|---|---:|
| Beginner | 30,000 |
| Intermediate | 70,000 |
| Advanced | 70,000 |
| Expert | 30,000 |

Difficulty should represent engineering complexity, not merely code length.

---

# 19. Prompt Style Distribution

Approximate target:

| Style | Percentage |
|---|---:|
| Very short / vague | 20% |
| Normal developer request | 40% |
| Detailed request | 30% |
| Highly constrained / expert | 10% |

Examples:

```text
Fix this.
```

```text
Refactor this.
```

```text
Create a cache.
```

```text
Create an async cache with TTL expiration and a maximum size.
```

```text
Implement a bounded async request scheduler using structured concurrency. Cancellation must propagate, concurrency must never exceed 20 tasks, and shutdown must not leak pending coroutines.
```

---

# 20. Domain Coverage

The dataset should cover Python broadly.

## 20.1 Core Python

- functions/classes
- dataclasses/enums
- protocols/ABCs
- generics/TypeVar
- advanced typing
- closures/decorators
- generators/iterators
- comprehensions
- context managers
- descriptors/properties
- exceptions
- pathlib
- collections/itertools/functools
- datetime
- regex
- JSON/CSV/serialization
- subprocess
- logging/configuration/environment
- pattern matching

## 20.2 Software Design

- cohesion/coupling
- dependency injection
- composition
- repository/service/factory/strategy/adapter
- layered/clean architecture
- domain boundaries
- DRY/KISS/YAGNI
- immutability
- testability

## 20.3 SOLID

- SRP
- OCP
- LSP
- ISP
- DIP

Include cases where applying SOLID helps and cases where extra abstractions would over-engineer the solution.

## 20.4 API Engineering

- REST
- HTTP semantics/status codes
- validation
- response schemas
- error contracts
- pagination/cursors
- filtering/sorting/search
- idempotency
- versioning
- authentication/authorization
- rate limiting
- retries/timeouts
- API clients
- webhooks
- middleware
- DI
- lifespan/background tasks
- readiness/health checks

Libraries may include FastAPI, Flask, Django, Starlette, Pydantic, requests, httpx, and unfamiliar supplied APIs.

## 20.5 Database Engineering

- modeling/normalization
- justified denormalization
- PK/FK/unique/check constraints
- indexes/composite indexes/index ordering
- relationships
- transactions
- isolation/locking
- optimistic/pessimistic concurrency
- pools/session lifecycle
- migrations
- query construction
- N+1
- pagination
- bulk operations
- upsert
- integrity handling
- repository/UoW

Libraries may include SQLAlchemy, Alembic, sqlite3, asyncpg, drivers, and unfamiliar supplied APIs.

## 20.6 Async / Concurrency / Parallelism

- asyncio
- TaskGroup/gather
- queues/semaphores/locks/events
- cancellation/timeouts
- backpressure
- producer-consumer
- threading
- multiprocessing
- futures
- races/deadlocks
- thread/process safety
- CPU-bound vs IO-bound
- graceful shutdown

## 20.7 Reliability

- retries
- retry classification
- exponential backoff/jitter
- circuit-breaker concepts
- timeout budgets
- fallback
- idempotency
- partial failure
- cleanup
- graceful degradation
- structured logging

## 20.8 Security

- injection
- path traversal
- SSRF
- unsafe deserialization
- authentication
- authorization
- secret handling
- password hashing concepts
- token validation
- least privilege
- file safety
- subprocess safety
- input validation

## 20.9 Testing

- pytest
- fixtures
- parametrization
- monkeypatching/mocks
- integration tests
- async tests
- regression tests
- property-oriented tests when useful
- testability-oriented refactoring

## 20.10 Performance

- complexity
- avoiding copies
- batching
- caching
- vectorization
- streaming
- lazy iteration
- bounded concurrency
- profiling-driven optimization

## 20.11 Data Science

- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- preprocessing
- missing values
- vectorization
- grouping/joins
- feature engineering
- model evaluation

## 20.12 Machine Learning

- PyTorch
- Dataset/DataLoader
- training/evaluation loops
- inference
- checkpointing
- device management
- gradients
- mixed precision concepts
- reproducibility

## 20.13 Data Engineering

- ETL
- ingestion
- batching
- streaming
- deduplication
- CSV/JSON/Parquet-oriented workflows
- validation
- chunked processing

## 20.14 Networking / Systems

- sockets
- HTTP clients
- retries
- connection pooling
- subprocess/signals
- filesystem
- temporary files
- OS interaction
- CLI tools

## 20.15 Packaging / Application Structure

- modules/packages
- imports
- dependency boundaries
- config/logging
- CLI
- application lifecycle

---

# 21. New-Library Generalization Data

This is a required dataset category.

Goal:

> Teach the model to learn enough of an unfamiliar API from context to integrate it correctly.

Use both real unfamiliar APIs and synthetic libraries. Synthetic libraries are valuable because they guarantee the base model cannot simply rely on memorization from pretraining.

Recommended target:

```text
10–20% of examples contain meaningful library_context
```

Include:

- known library + explicit docs
- less-common library
- version-changed API
- synthetic unseen package
- custom internal-style SDK
- API client SDK
- database SDK
- cache SDK
- queue SDK
- storage SDK
- auth SDK

The context should teach API surface. The model must still provide engineering judgment.

## 21.1 New Library Example — Cache SDK

Library context:

```text
Package: acmecache

class CacheClient:
    def __init__(self, url: str): ...

    async def get(self, key: str) -> bytes | None: ...

    async def set(
        self,
        key: str,
        value: bytes,
        ttl: int
    ) -> None: ...

    async def delete(self, key: str) -> None: ...
```

Instruction:

```text
Build an async user repository that loads a user from the database and caches the serialized user for 60 seconds.
```

Expected behavior:

- inject/create `CacheClient`
- await async methods
- cache lookup first
- safe serialization/deserialization
- DB fallback
- set TTL 60
- maintain testability

## 21.2 New Library Example — HTTP SDK

Context:

```text
Package: orbit_http

class Client:
    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict | None = None,
        timeout: float = 10.0
    ) -> Response: ...

class Response:
    status_code: int
    def json(self) -> dict: ...
    def raise_for_status(self) -> None: ...
```

Instruction:

```text
Create a typed async client for the Orders API. Retry 429 and 503 responses up to three times with exponential backoff. Do not retry other 4xx errors.
```

## 21.3 New Library Example — Database SDK

Context:

```text
Package: nova_db

class Database:
    async def transaction(self) -> AsyncContextManager[Transaction]: ...

class Transaction:
    async def fetch_one(self, query: str, params: dict) -> Row | None: ...
    async def execute(self, query: str, params: dict) -> None: ...
```

Instruction:

```text
Implement an atomic balance transfer between two accounts.
```

The model must combine unfamiliar API usage with transaction semantics and correctness knowledge.

---

# 22. Dataset Examples

## 22.1 Generate Example

```json
{
  "id": "pys_gen_000001",
  "task": "generate",
  "instruction": "Create an async TTL cache with a maximum size.",
  "code_before": "",
  "error": "",
  "project_context": "",
  "library_context": "",
  "tests": "pytest code here",
  "target_mode": "build",
  "target_ir": "<BUILD> ... <END>",
  "target_code": "actual Python source",
  "domain": "core_python",
  "subdomain": "caching",
  "difficulty": "intermediate",
  "instruction_style": "normal",
  "concepts": ["asyncio", "ttl", "eviction"],
  "libraries": ["asyncio"],
  "verified": true,
  "verification_level": "tests",
  "source_type": "teacher_synthetic",
  "source_dataset": "qwen-coder-teacher",
  "source_reference": "teacher:generate:000001",
  "ir_version": "v1",
  "content_hash": "..."
}
```

## 22.2 Repair Example

Instruction:

```text
Fix this.
```

Code before:

```python
async def fetch(client, url):
    return client.get(url)
```

Error:

```text
RuntimeWarning: coroutine was never awaited
```

Target IR:

```text
<EDIT>
<REPLACE>
target fetch.return.value
<AWAIT>
<CALL>
function client.get
arg url
<END_CALL>
</AWAIT>
<END_REPLACE>
<END>
```

Target code:

```python
async def fetch(client, url):
    return await client.get(url)
```

## 22.3 Refactor Example

Instruction:

```text
Refactor this.
```

Code before:

```python
db = ...
mailer = ...

class UserService:
    def create(self, email):
        user = db.insert(email)
        mailer.send(email, "welcome")
        return user
```

Possible target:

```python
class UserService:
    def __init__(self, repository, notifier):
        self._repository = repository
        self._notifier = notifier

    def create(self, email):
        user = self._repository.create(email)
        self._notifier.send_welcome(user)
        return user
```

The model should decide whether dependency injection is justified by surrounding context.

---

# 23. IR Examples

## 23.1 Build Function

```text
<BUILD>

<FUNCTION>
name unique_values

<PARAM>
name values
type Iterable[str]
<END_PARAM>

<RETURN_TYPE>
list[str]
<END_RETURN_TYPE>

<ASSIGN>
target seen
value <CALL> set <END_CALL>
<END_OP>

<ASSIGN>
target result
value []
<END_OP>

<FOR>
target value
iter values

<IF>
condition value not in seen

<CALL>
function seen.add
arg value
<END_CALL>

<CALL>
function result.append
arg value
<END_CALL>

<END_IF>
<END_FOR>

<RETURN>
value result
<END_RETURN>

<END_FUNCTION>
<END>
```

## 23.2 Library-Specific Usage Without Library-Specific Compiler

Input asks for Flask.

IR:

```text
<IMPORT_FROM>
module flask
names Blueprint, jsonify
<END_OP>

<ASSIGN>
target bp
value <CALL> Blueprint args "users", __name__ <END_CALL>
<END_OP>

<FUNCTION>
name list_users

<DECORATOR>
<CALL>
function bp.get
arg "/users"
<END_CALL>
<END_DECORATOR>

<RETURN>
<CALL>
function jsonify
arg users
<END_CALL>
<END_RETURN>

<END_FUNCTION>
```

Compiler only understands imports, assignments, calls, decorators, functions and returns. It does not understand Flask semantics.

---

# 24. Dataset Generation Strategy

Do not manually generate 200K examples.

Use a resumable dataset factory:

```text
coverage controller
        ↓
public data + teacher synthesis
        ↓
semantic candidate
        ↓
parse / compile
        ↓
tests / behavior validation
        ↓
IR derivation
        ↓
deduplication
        ↓
quality filtering
        ↓
accepted dataset
```

Build a candidate pool larger than the final dataset.

Example:

```text
400K–600K candidates
        ↓
validation/filtering
        ↓
best balanced 200K
```

---

# 25. Teacher-Model Strategy

A stronger Qwen coding model can act as the offline teacher.

This is sequence-level knowledge distillation.

Teacher responsibilities:

- realistic instructions
- correct target code
- realistic broken code
- realistic refactor candidates
- tests
- error scenarios
- engineering decisions
- unfamiliar-library integration examples

Teacher should not be trusted blindly.

Everything possible should be checked deterministically.

Teacher does not need to generate:

- content hashes
- structural hashes
- AST
- simple deterministic metadata

For the final IR, prefer deterministic derivation from verified code wherever possible. If a high-level operation cannot be inferred mechanically, use a controlled semantic annotation step and validate it against the source/target program.

---

# 26. Deterministic IR Construction

For build tasks:

```text
target_code
    ↓
ast.parse
    ↓
AST/CST analyzer
    ↓
normalize into IR grammar
    ↓
target_ir
```

For repair/refactor:

```text
code_before ──→ parse
                   │
                   ├── structural diff
                   │
target_code ───→ parse
                   │
                   ↓
             edit-oriented IR
```

Avoid representations based only on node counts.

Bad:

```text
Name +10
Call +5
Assign +3
```

Good:

```text
<REPLACE>
target fetch.return.value
value <AWAIT> ... </AWAIT>
</REPLACE>
```

Possible edit operations:

```text
<INSERT>
<DELETE>
<REPLACE>
<MOVE>
<RENAME>
<CHANGE_SIGNATURE>
```

If an edit representation becomes more complicated than rebuilding the program, allow a normalized full-program replacement mode explicitly.

---

# 27. Verification Rules

`verified=true` must have a precise meaning.

Recommended levels:

## `structural`

```text
target_code parses
target_code compiles
IR parses
IR compiles to structurally equivalent Python
```

## `tests`

```text
structural checks
+
provided tests pass
```

## `behavioral`

Behavioral invariants are checked.

## `static`

Structural/tests plus selected lint/type/static checks.

## `full`

Highest practical verification available for that example.

### Generate

```text
target_code
→ parse PASS
→ compile PASS
→ tests PASS
```

### Full-File Generate

Same plus complete importable module and no omitted sections.

### Repair

Strongest form:

```text
code_before + tests → FAIL
target_code + tests → PASS
```

A powerful synthetic strategy:

```text
good target
    ↓
deterministic bug mutation
    ↓
confirm tests fail
    ↓
broken version becomes code_before
```

### Refactor

Strongest form:

```text
code_before + tests → PASS
target_code + same tests → PASS
```

---

# 28. Deduplication and Contamination Control

Use multiple layers.

## Exact Content Hash

Normalize and hash instruction, code_before and target_code.

## AST-Normalized Hash

Normalize formatting and selected identifiers/literals, then hash structure.

## Token / MinHash Similarity

Catch copied examples with small changes.

## Embedding Similarity

Use for suspicious clusters or later-stage filtering if compute permits.

## Benchmark Contamination

Do not intentionally include exact benchmark problems/solutions from future evaluation sets such as:

- HumanEval
- HumanEval+
- MBPP
- MBPP+
- DS-1000
- BigCodeBench

Reject public-source examples that closely match known evaluation tasks.

---

# 29. Training Input and Target Format

Use stable section tags.

Example input:

```text
<TASK>
repair
</TASK>

<INSTRUCTION>
Fix this.
</INSTRUCTION>

<PROJECT_CONTEXT>
Python 3.12
Existing application is async.
</PROJECT_CONTEXT>

<LIBRARY_CONTEXT>
httpx.AsyncClient.get(url, ...) -> Response
</LIBRARY_CONTEXT>

<CODE_BEFORE>
async def fetch(client, url):
    return client.get(url)
</CODE_BEFORE>

<ERROR>
RuntimeWarning: coroutine was never awaited
</ERROR>
```

Target:

```text
<EDIT>
...
<END>
```

Not every section must contain data, but serialization rules must remain consistent between training and inference.

---

# 30. Inference Pipeline

```text
user request
    ↓
collect project context
    ↓
collect relevant library context when available
    ↓
serialize input
    ↓
CodeT5+ encoder
    ↓
structured decoder
    ↓
IR grammar constrained decoding
    ↓
IR
    ↓
deterministic IR compiler
    ↓
AST/CST
    ↓
Python source
    ↓
verification
    ↓
PASS ─────────────→ output
    │
    └─ FAIL
         ↓
verification error + previous output
         ↓
repair pass
```

The user normally sees Python source, not the IR.

---

# 31. Evaluation Plan

Evaluation must cover more than token similarity.

## 31.1 Syntax / Structure

- IR validity rate
- source parse rate
- source compile rate

## 31.2 Functional Correctness

- unit-test pass rate
- hidden-test pass rate
- repair success rate
- refactor behavior preservation

## 31.3 Engineering Quality

Evaluate targeted dimensions separately:

- typing
- error handling
- API design
- DB correctness
- async correctness
- auth/authz
- security
- resource management
- design simplicity
- maintainability

## 31.4 New-Library Generalization

Hold out synthetic libraries entirely from training.

At evaluation time provide only docs/signatures/examples and measure whether the model integrates them correctly.

## 31.5 User-Constraint Following

Evaluate:

- explicit framework
- explicit library
- Python version
- sync/async requirement
- forbidden dependency
- performance constraint
- security constraint

## 31.6 Project-Context Following

Example:

- project uses Flask
- user asks for a new endpoint
- model should not unexpectedly migrate to FastAPI

---

# 32. Ablation Plan

Test the architecture scientifically.

## A. Direct Python Baseline

```text
input → target_code
```

## B. IR Without Grammar Constraint

```text
input → target_ir
```

## C. IR + Grammar Constraint

```text
input → constrained target_ir
```

## D. IR + Library Context

Adds docs/signatures.

## E. IR + Verification Repair

Adds execution/static feedback loop.

## F. IR + Explicit Symbol/Type Features

If later architecture modifications permit.

Compare:

- functional correctness
- syntax validity
- parameter count
- decoding speed
- new-library generalization
- repair accuracy
- refactor behavior preservation

---

# 33. Scaling Plan

Do not immediately generate 200K rows.

Recommended progression:

```text
Phase 0: 20–100 examples
validate schema + IR manually

Phase 1: 400 examples
100 per task

Phase 2: 2,000 examples
coverage and verification audit

Phase 3: 10,000 examples
small fine-tuning experiment

Phase 4: 50,000 examples
measure architecture viability

Phase 5: candidate pool 400K+
filter/balance

Phase 6: final 200K accepted dataset
```

At every stage inspect:

- rejection rate
- duplicate rate
- teacher failure patterns
- IR compiler failures
- test quality
- difficulty calibration
- domain balance
- library-context quality
- code quality

---

# 34. Open Research Questions

These should remain explicit rather than being prematurely fixed.

## 34.1 How High-Level Should the IR Be?

Too low-level:

```text
raw AST
```

wastes tokens.

Too high-level:

```text
use repository pattern
```

is not deterministic enough to compile.

The IR must capture enough implementation detail to reconstruct actual Python.

## 34.2 Build vs Edit IR

For Repair/Refactor, compare:

- editing existing AST
- rebuilding full target IR
- choosing dynamically

## 34.3 Comments and Formatting

AST alone does not preserve comments.

Potential solution:

- explicit comment/docstring IR nodes
- CST renderer
- formatting hints

## 34.4 Library Context Compression

Large libraries contain too many APIs.

Future component:

```text
library docs
    ↓
retrieval
    ↓
relevant signatures/examples
    ↓
encoder
```

## 34.5 Pretrained Knowledge Retention

Fine-tuning should specialize the model without destroying broad Python knowledge. Broad held-out evaluation is required to detect catastrophic specialization.

## 34.6 One Decoder vs Two-Stage Neural Design

First design:

```text
one encoder + one structured decoder
```

Possible later experiment:

```text
design decoder
    ↓
implementation decoder
```

Do not add this complexity before the single-decoder architecture is tested.

---

# 35. Final Design Summary

```text
Base model:
Salesforce/codet5p-220m-py

Capabilities:
Generate
Full-file Generate
Repair
Refactor

Input:
requirement
existing code
error/traceback
project context
library/API context
tests when appropriate

Neural responsibility:
understand
reason
design
select libraries
respect explicit constraints
use existing project conventions
integrate unfamiliar APIs
choose architecture
choose algorithms
choose patterns only when useful
design APIs
design DB behavior
security/auth/authz
async/concurrency
typing
errors
calls/imports/control flow

Decoder output:
formal Python-oriented IR

Deterministic responsibility:
IR grammar validation
IR → AST/CST
source rendering
parse/compile/tests/static verification

New-library strategy:
supply relevant docs/signatures/examples to encoder
model learns API integration
compiler remains library-agnostic

Primary dataset:
200K mature target
50K per task
balanced difficulty
realistic prompt styles
broad Python domains
10–20% meaningful library-context examples
strong verification
multi-layer deduplication

Training:
CodeT5+ 220M Python
input → structured IR

User-visible output:
Python source
```

The central principle is:

> **The model owns software intelligence. The deterministic layer owns syntax mechanics.**

The main research question is:

> **Can a compact pretrained encoder-decoder retain broad Python engineering knowledge and generalize better by decoding a constrained Python-oriented program representation rather than raw source text?**
