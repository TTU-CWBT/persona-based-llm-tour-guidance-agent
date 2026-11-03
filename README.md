# Persona-Based LLM Agent Architecture for Tour Guiding Tasks

This repository contains the current public-facing Python implementation of a persona-based LLM agent architecture for tour guiding tasks. The runtime uses Ollama as the LLM backend and separates persona integration, task continuity, planning/routing, internal/external retrieval, observation evaluation, response-scope construction, and conversation memory into modules.

## Repository structure

```text
.
├── CLP.py
├── memory_repository.py
├── persona.json
├── environment.yml
├── requirements.txt
├── LAB_KB_GUIDE.md
├── PERSONA_GUIDE.md
├── PERSONA_GPT_PROMPT.md
├── personas/
│   ├── persona_template.json
│   ├── tsubomi.json
│   └── majima-ritsu.json
├── backends/
│   ├── backend_base.py
│   └── backend_ollama.py
├── features/
│   ├── agent_executor.py
│   ├── agent_planner.py
│   ├── agent_state.py
│   ├── agent_trace.py
│   ├── extkb.py
│   ├── labkb.py
│   ├── memory_wrap.py
│   ├── observation_evaluator.py
│   ├── observation_judge.py
│   ├── persona.py
│   ├── progress.py
│   ├── project_context.py
│   ├── task_continuity_resolver.py
│   ├── task_scope_builder.py
│   └── webtools.py
└── lab_kb/
    ├── build_lab_repo.py
    ├── lab_kb_config.yaml
    ├── lab_repository.py
    ├── raw/
    └── store/
```

## Environment

The included `environment.yml` is based on the supplied Conda export with the machine-specific `prefix:` removed.

```bash
conda env create -f environment.yml
conda activate llm
```

Alternatively, install the direct runtime dependencies with:

```bash
pip install -r requirements.txt
```

Ollama must be installed separately, and the selected model must already exist in Ollama.

## External search configuration

Copy `.env.example` to `.env` and provide a Tavily key only if Tavily search is required:

```text
TAVILY_API_KEY=your_key_here
```

The real key must never be committed to Git.


## Create and use custom personas

Named persona files can be placed under `personas/`. Two example personas and one neutral template are included:

```text
personas/tsubomi.json
personas/majima-ritsu.json
personas/persona_template.json
```

To use a named persona:

```bash
python CLP.py --cli --persona tsubomi --ollama_model <your-ollama-model>
```

To create your own persona, either edit `personas/persona_template.json` or use the ready-to-copy ChatGPT prompt in **[PERSONA_GPT_PROMPT.md](PERSONA_GPT_PROMPT.md)**. See **[PERSONA_GUIDE.md](PERSONA_GUIDE.md)** for the runtime-supported fields, design recommendations, validation checklist, and persona-switching workflow.

## Prepare the internal Lab KB

Put private/internal source documents under:

```text
lab_kb/raw/
```

Then build the index from the project root:

```bash
python lab_kb/build_lab_repo.py --rebuild
```

For later additions of new files, run:

```bash
python lab_kb/build_lab_repo.py
```

See **[LAB_KB_GUIDE.md](LAB_KB_GUIDE.md)** for the complete workflow, supported file types, indexing behavior, runtime integration, and Git safety notes.

## Run

CLI:

```bash
python CLP.py --cli --ollama_model <your-ollama-model>
```

Gradio:

```bash
python CLP.py --gradio --ollama_model <your-ollama-model>
```

`persona.json` is retained because the current `CLP.py` loads it by default when `--persona` is not specified.
