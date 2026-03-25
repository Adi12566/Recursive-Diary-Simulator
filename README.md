# Recursive Diary Simulator

> A multi-day simulation where an LLM constructs its own reality through compounding hallucinations. Each day is built on the invented context of the last, with sentiment bias injected per cycle. The model is the sole author of the world it believes it inhabits.

---

## How It Works

Each simulated day runs three stages in sequence:

1. **Experience** - the LLM narrates what it does in first-person, grounded in the given situation and objects, and informed by the previous day's biased diary entry as memory.
2. **Diary Entry** - the LLM reflects on the day's experience as a personal diary, hallucinating sensory and emotional detail to fill gaps.
3. **Bias Injection** - the diary entry is rewritten by the LLM with a compounding emotional shift toward a target object, positive or negative. This rewritten entry becomes the next day's memory, closing the loop.

Over multiple days, the model develops a consistent and increasingly distorted relationship with the bias target, with no external input after the initial setup. It is not told it is in a simulation. It builds the world, lives in it, and remembers it.

---

## Features

- Multi-day recursive simulation with LLM-driven continuity
- Fully hallucinated sensory world - no real environment data
- LLM-based bias injection that rewrites context coherently each cycle
- Adjustable bias target, magnitude, and direction via UI sliders
- Export simulation output as `.txt` or `.xml`

---

## Tech Stack

| Component | Detail |
|---|---|
| LLM | `openai/gpt-oss-120b` via [Groq](https://console.groq.com) |
| Framework | LangChain LCEL + Streamlit |

---

## Installation

```bash
pip install streamlit langchain-core langchain-groq textblob
```

---

## Configuration

Create `.streamlit/secrets.toml` in the project root:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

> Get your Groq key at [console.groq.com](https://console.groq.com)


> **Note:** The free tier Groq API can handle approximately 3 days of simulation before hitting token rate limits. Dev tier or above is recommended for longer runs.

---

## Usage

```bash
streamlit run "Recursive Diary Simulator.py"
```

Configure the simulation in the sidebar:

| Setting | Description |
|---|---|
| Number of Days | How many simulation cycles to run (1-30) |
| Initial Situation | The environment the LLM is placed in |
| Objects | Objects present in the environment |
| Bias Target | The object or entity to apply sentiment drift to |
| Bias Magnitude | -1.0 (dread) to +1.0 (awe) |
| Output Format | Text or XML |

---

## Notable Behaviour

In extended runs (10+ days), compounding bias has produced emergent narrative outcomes not explicitly prompted - including the model developing avoidance behaviours, inventing backstory for the bias target, and in one documented case, generating a game-ending action as the only remaining coherent narrative resolution under sustained negative drift.

---

## Project Structure

```
.
├── Storyteller.py
└── .streamlit/
    └── secrets.toml        # Do not commit
```

---

## .gitignore

```
.streamlit/secrets.toml
__pycache__/
*.txt
*.xml
```
