# Persona Creation Guide / 人格建立指南

This project supports named persona files under `personas/`. A persona can be created manually or generated with ChatGPT, then loaded without changing the Python source code.

本專案支援將人格設定放在 `personas/` 資料夾中。你可以手動建立，也可以直接請 ChatGPT 依照本專案格式產生 JSON，之後不需要修改 Python 程式即可套用。

## Included examples / 內附範例

- `personas/tsubomi.json`: a warm, reliable, slightly playful lab/research assistant persona.
- `personas/majima-ritsu.json`: a restrained, low-temperature, control-oriented fictional persona.
- `personas/persona_template.json`: a neutral template containing the fields currently supported by the runtime persona loader.

The two sample personas demonstrate that the same architecture can express very different personalities while keeping the same JSON structure.

## Quick start / 快速使用

1. Copy `personas/persona_template.json` or ask ChatGPT to generate a compatible persona.
2. Save the result as a JSON file under `personas/`, for example:

```text
personas/my_guide.json
```

3. Start the agent from the project root:

```bash
python CLP.py --cli --persona my_guide --ollama_model <your-ollama-model>
```

`--persona my_guide` maps to `./personas/my_guide.json`. Do not include `.json` in the command-line argument.

If `--persona` is omitted, the current program falls back to the root-level `persona.json`.

## Runtime-supported persona fields / 目前程式實際使用的欄位

The current `features/persona.py` actively consumes the following rich-persona fields:

- `name`
- `role_summary`
- `basic_profile.main_context`
- `beliefs`
- `personality_core.big_five`
  - `extraversion`
  - `conscientiousness`
  - `agreeableness`
  - `openness`
  - `neuroticism`
  - each trait may contain `core_tendency` and `adaptive_expression`
- `personality_core.honesty_humility`
  - `core_tendency`
  - `ethical_sensitivity`
- `emotion_model.baseline`
  - `default`
  - `familiar_context`
  - `unfamiliar_context`
- `emotion_model.triggers`
  - `target_group`
  - `trigger_condition`
  - `inner_shift`
  - `outward_response`
  - `response_intensity`
- `linguistic_habits`
  - `core_patterns`
  - `preferred_markers`
  - `avoid_markers`
  - `sentence_style`
- `interaction_profiles.lab`
- `interaction_profiles.student`
- `interaction_behaviors.addressing_rules`
- `interaction_behaviors.first_reply`
- `interaction_behaviors.tone_formation_rules` (optional)
- `banned_phrases` (optional)

Additional JSON fields are allowed, but fields not read by `features/persona.py` act only as metadata unless the code is extended to use them.

For example, the sample personas contain design/documentation fields such as `theoretical_basis` and `migration_notes`. These are useful for explaining how a persona was designed, but they are not directly injected into the runtime prompt by the current implementation.

Likewise, `majima-ritsu.json` contains `interaction_behaviors.threat_presence_rules`. That field documents the persona design, but the current runtime does not read it directly. If a custom persona needs an instruction to affect runtime behavior, place the same idea in a supported field such as `linguistic_habits`, `emotion_model`, `interaction_profiles`, or `tone_formation_rules`.

## Recommended design principle / 建議設計原則

A useful persona should distinguish stable personality from audience adaptation.

- Put stable personality traits in `personality_core`.
- Put value judgments and recurring priorities in `beliefs`.
- Put default affect and situation-triggered changes in `emotion_model`.
- Put wording, rhythm, and recurring expression patterns in `linguistic_habits`.
- Put audience-dependent presentation differences in `interaction_profiles`.
- Put forms of address and self-introduction behavior in `interaction_behaviors`.

Avoid describing the same behavior independently in many fields. If every field repeats the same sentence, the persona becomes harder to maintain and may over-constrain the model.

## Creating a persona with ChatGPT / 用 ChatGPT 建立人格

A ready-to-copy prompt is provided in `PERSONA_GPT_PROMPT.md`.

Recommended workflow:

1. Tell ChatGPT the character or personality you want.
2. Describe important traits, values, emotional style, relationships, speaking habits, and intended users.
3. Paste the prompt from `PERSONA_GPT_PROMPT.md`.
4. Ask ChatGPT to output JSON only.
5. Save the result in `personas/<name>.json`.
6. Run the program with `--persona <name>`.

## Validation checklist / 儲存前檢查

Before using a generated persona:

- The file must be valid JSON. JSON does not allow comments or trailing commas.
- Use double quotes around keys and strings.
- Keep `name` and `role_summary` non-empty.
- Keep all five Big Five traits if you want the full persona structure.
- Make emotional triggers concrete enough that the model can recognize when they apply.
- Avoid giving two fields directly contradictory instructions.
- Do not put API keys, passwords, private personal data, or machine-specific paths in persona files.
- Treat model-generated persona content as a draft: read it once before using it.

A quick JSON syntax check can be performed with:

```bash
python -m json.tool personas/my_guide.json > nul
```

On macOS/Linux:

```bash
python -m json.tool personas/my_guide.json > /dev/null
```

## Switching personas / 切換人格

No Python file needs to be edited when switching between named personas.

```bash
python CLP.py --cli --persona tsubomi --ollama_model <your-ollama-model>
python CLP.py --cli --persona majima-ritsu --ollama_model <your-ollama-model>
python CLP.py --cli --persona my_guide --ollama_model <your-ollama-model>
```

The conversation-memory layer also derives a memory profile from the active persona, helping different named personas avoid sharing the same runtime conversation state.
