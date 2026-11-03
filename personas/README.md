# Personas

This folder contains named persona configurations for the agent runtime.

Included files:

- `persona_template.json` — neutral runtime-compatible template.
- `tsubomi.json` — example of a warm, dependable, slightly playful assistant persona.
- `majima-ritsu.json` — example of a restrained, low-temperature fictional persona with a strong sense of control and boundaries.

To load a persona, use the file name without `.json`:

```bash
python CLP.py --cli --persona tsubomi --ollama_model <your-ollama-model>
```

For the full schema and design guidance, see [`../PERSONA_GUIDE.md`](../PERSONA_GUIDE.md).

To generate a compatible persona with ChatGPT, copy the prompt in [`../PERSONA_GPT_PROMPT.md`](../PERSONA_GPT_PROMPT.md).
