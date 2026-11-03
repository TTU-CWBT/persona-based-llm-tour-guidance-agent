# ChatGPT Prompt for Creating a Compatible Persona

Copy the prompt below into ChatGPT, then add the character/personality description you want at the bottom.

## 中文版（建議）

```text
我正在使用一套 Persona-Based LLM Agent 程式。請根據我接下來提供的人格需求，產生一份可以直接儲存為 JSON 並放入 `personas/` 資料夾使用的人格設定。

請遵守以下要求：

1. 最終只輸出一個合法 JSON object，不要使用 Markdown code block，不要在 JSON 前後加入解釋。
2. JSON 必須使用雙引號，不能有註解、不能有 trailing comma。
3. 請使用下列主要結構，不要自行改名：
   - name
   - role_summary
   - basic_profile.main_context
   - beliefs
   - personality_core.big_five
     - extraversion
     - conscientiousness
     - agreeableness
     - openness
     - neuroticism
     - 每個 trait 使用 core_tendency、adaptive_expression
   - personality_core.honesty_humility
     - core_tendency
     - ethical_sensitivity
   - emotion_model.baseline
     - default
     - familiar_context
     - unfamiliar_context
   - emotion_model.triggers
     - id
     - target_group
     - trigger_condition
     - inner_shift
     - outward_response
     - response_intensity
   - linguistic_habits
     - core_patterns
     - preferred_markers
     - avoid_markers
     - sentence_style
   - interaction_profiles.lab
   - interaction_profiles.student
     - target
     - tone
     - warmth
     - humor_level
     - directness
     - pressure_reduction
     - clarity
     - example_density
     - rewriting_preference
     - examples
   - interaction_behaviors.addressing_rules
   - interaction_behaviors.first_reply
     - enabled
     - triggers
     - template
     - call_sign
   - interaction_behaviors.tone_formation_rules（可選）
   - banned_phrases（可選）

4. Big Five 描述的是穩定人格傾向，不要把單一場景反應寫成人格特質。
5. Honesty–Humility 用來補充誠實、謙遜、操控、利益、公平、信任等價值傾向；不要把它寫成硬性的系統安全政策。
6. emotion_model.baseline 描述角色平時的情緒基線；triggers 描述特定情境發生時，內在如何改變，以及外在如何表現。
7. linguistic_habits 要描述可觀察的對話習慣，例如句長、節奏、慣用表達、避免表達；不要只寫「自然一點」「有個性」這種無法執行的敘述。
8. interaction_profiles 只處理面對不同對象時的語用差異，不要在這裡重新定義整套人格。
9. 不要讓不同欄位互相矛盾。如果需求本身有矛盾，請選擇最合理且能維持角色一致性的解釋。
10. 不要加入我沒有要求的真實個資、API key、檔案路徑、研究成果或系統能力。
11. 請讓每一欄都有足夠具體的描述，使語言模型能實際模仿，但不要用大量同義句重複同一個特質。
12. 如果角色帶有威脅感、敵意、黑色幽默或其他強烈風格，可以保留人格表現，但不要加入現實危害行為的操作細節。

以下是我想建立的人格：

【角色名稱】
（填寫）

【角色定位／用途】
（填寫）

【核心人格印象】
（填寫）

【重視的價值或信念】
（填寫）

【情緒基線】
（填寫）

【容易被觸發的情境與反應】
（填寫）

【說話方式／口頭習慣】
（填寫）

【對熟悉對象的互動方式】
（填寫）

【對學生、陌生人或外部使用者的互動方式】
（填寫）

【希望避免的說話方式】
（填寫）

【其他補充】
（填寫）
```

## English version

```text
I am using a Persona-Based LLM Agent project. Based on the persona description I provide below, generate a persona configuration that can be saved directly as JSON under the project's `personas/` directory.

Requirements:

1. Output exactly one valid JSON object. Do not use a Markdown code block and do not add explanations before or after the JSON.
2. Use double quotes. Do not use comments or trailing commas.
3. Use these runtime-compatible fields:
   - name
   - role_summary
   - basic_profile.main_context
   - beliefs
   - personality_core.big_five
     - extraversion
     - conscientiousness
     - agreeableness
     - openness
     - neuroticism
     - each trait should contain core_tendency and adaptive_expression
   - personality_core.honesty_humility
     - core_tendency
     - ethical_sensitivity
   - emotion_model.baseline
     - default
     - familiar_context
     - unfamiliar_context
   - emotion_model.triggers
     - id
     - target_group
     - trigger_condition
     - inner_shift
     - outward_response
     - response_intensity
   - linguistic_habits
     - core_patterns
     - preferred_markers
     - avoid_markers
     - sentence_style
   - interaction_profiles.lab
   - interaction_profiles.student
   - interaction_behaviors.addressing_rules
   - interaction_behaviors.first_reply
   - interaction_behaviors.tone_formation_rules (optional)
   - banned_phrases (optional)
4. Treat Big Five as stable personality tendencies, not one-off situational reactions.
5. Use Honesty-Humility for values involving sincerity, fairness, modesty, manipulation, self-interest, and trust; do not turn it into a hard system safety policy.
6. Use emotion_model.baseline for the default affective style and triggers for recognizable situation-dependent changes.
7. Make linguistic_habits observable and actionable: sentence rhythm, recurring expressions, wording preferences, and styles to avoid.
8. Use interaction_profiles only for audience-dependent pragmatic adaptation. Do not redefine the whole personality there.
9. Avoid contradictions and unnecessary repetition across fields.
10. Do not invent private personal data, API keys, local file paths, research achievements, or system capabilities.
11. Keep every field concrete enough for an LLM to imitate.
12. Strong fictional styles are allowed, but do not add operational instructions for real-world harm.

Persona I want:

Name:
[fill in]

Role / intended use:
[fill in]

Core personality impression:
[fill in]

Values / beliefs:
[fill in]

Emotional baseline:
[fill in]

Important triggers and reactions:
[fill in]

Speaking habits:
[fill in]

Behavior toward familiar users:
[fill in]

Behavior toward students / unfamiliar users:
[fill in]

Styles or phrases to avoid:
[fill in]

Other notes:
[fill in]
```
