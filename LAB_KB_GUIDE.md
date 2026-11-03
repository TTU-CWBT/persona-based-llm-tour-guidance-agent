# Lab KB 使用說明

本專案的 `lab_kb/` 用於保存與檢索「內部知識資料」，例如實驗室研究資料、系統說明文件、論文、Markdown、文字檔或 CSV。它與 `memory_repository.py` 的對話記憶不同：

- `lab_kb/`：長期存在、由文件建立的內部知識庫，透過 BGE-M3 + FAISS（或 sklearn）進行向量檢索。
- `memory_repository.py`：保存使用者與代理人對話、摘要、task 與 open loop，用於維持對話脈絡。

如果你的目的，是讓導覽代理人「多知道一份內部資料」，應把資料加入 `lab_kb/raw/`，再重建或增量更新索引。

## 1. 資料夾結構

```text
lab_kb/
├── build_lab_repo.py
├── lab_kb_config.yaml
├── lab_repository.py
├── raw/
│   └── .gitkeep
└── store/
    └── .gitkeep
```

`raw/` 放原始文件；`store/` 由程式自動生成索引，不需要手動編輯。

GitHub 版本預設不提交 `raw/` 和 `store/` 的實際內容，只保留 `.gitkeep`。這樣可以避免把私人實驗室資料、論文內容或已建立索引中的文字 chunk 一起公開。

## 2. 支援的原始資料格式

目前 `lab_kb_config.yaml` 設定支援：

```text
.txt
.md
.pdf
.csv
```

PDF 會使用 `pypdf` 擷取可選取的文字。如果 PDF 只有掃描影像而沒有文字層，目前程式不會自動 OCR。

## 3. 新增一份內部知識

例如要新增一份：

```text
lab_information.pdf
```

先把檔案放到：

```text
lab_kb/raw/lab_information.pdf
```

`raw/` 可以建立子資料夾，程式會遞迴搜尋，例如：

```text
lab_kb/raw/
├── papers/
│   ├── paper_a.pdf
│   └── paper_b.pdf
├── system/
│   └── system_overview.md
└── lab_information.pdf
```

## 4. 第一次建立 Lab KB

在專案根目錄執行：

```bash
python lab_kb/build_lab_repo.py --rebuild
```

程式會：

1. 讀取 `lab_kb/lab_kb_config.yaml`。
2. 掃描 `lab_kb/raw/` 中支援的文件。
3. 清理文字並依 `chunk_size` / `chunk_overlap` 分塊。
4. 使用設定的 embedding model 建立向量。
5. 建立 FAISS 索引；若環境沒有可用的 FAISS，則退回 sklearn NearestNeighbors。
6. 將結果寫入 `lab_kb/store/`。

目前會產生的主要檔案包含：

```text
lab_kb/store/
├── vectors.npy
├── meta.json
└── faiss.index       # 使用 FAISS 時產生
```

注意：`meta.json` 會包含分塊後的原始文字，因此不要把 `store/` 直接提交到公開 GitHub。

## 5. 只新增新文件時

如果索引已經存在，而且只是把新的文件放入 `raw/`，可以執行：

```bash
python lab_kb/build_lab_repo.py
```

沒有 `--rebuild` 時，程式會比較目前 `raw/` 中的檔案與已索引來源，只把尚未索引的新檔加入 repository。

例如：

```text
原本：
lab_kb/raw/paper_a.pdf

後來新增：
lab_kb/raw/paper_b.pdf
```

執行：

```bash
python lab_kb/build_lab_repo.py
```

程式只會新增 `paper_b.pdf` 的內容。

## 6. 什麼時候應使用 --rebuild

建議在以下情況完整重建：

```bash
python lab_kb/build_lab_repo.py --rebuild
```

- 修改既有原始文件內容。
- 刪除或替換已經建立索引的文件。
- 修改 embedding model。
- 修改 `chunk_size` 或 `chunk_overlap`。
- 想確保整個索引和目前 `raw/` 完全一致。

目前的增量模式主要用於「加入新檔案」，不負責自動同步已修改或已刪除的舊檔案。

## 7. Lab KB 設定

目前 `lab_kb_config.yaml` 包含：

```yaml
raw_dir: "./raw"
store_dir: "./store"
embedding_model: "BAAI/bge-m3"
normalize_embeddings: true
chunk_size: 800
chunk_overlap: 120
top_k: 8
min_score: 0.22
use_faiss: true
```

`build_lab_repo.py` 會以 YAML 檔案所在的 `lab_kb/` 目錄解析 `./raw` 與 `./store`，所以可以直接從專案根目錄執行建置指令。

注意：正式 runtime 的 `features/labkb.py` 目前另外建立自己的 `LabRepoConfig`，其中 retrieval 設定包含 `top_k=4`、`min_score=0.30`，並會先取得較多候選 chunk 再做 rerank、來源多樣化與 evidence handling。換句話說，YAML 是索引建置工具使用的設定；實際 agent runtime 的檢索策略以 `features/labkb.py` 為準。

## 8. 主程式如何使用 Lab KB

主程式不會直接呼叫 `lab_repository.py`。正式 runtime 流程是：

```text
CLP.py
  ↓
AgentPlanner
  ↓ knowledge_lookup
AgentExecutor
  ↓ retrieve_lab_kb
features/labkb.py
  ↓
lab_kb/lab_repository.py
  ↓
lab_kb/store/
```

`features/labkb.py` 會處理：

- query augmentation
- LabRepository 檢索
- reference-heavy chunk 降權
- 同來源限制
- 來源多樣化 rerank
- ownership signal
- structured observation

因此正式 agent 執行時，應走 `features/labkb.py`，不要自行從 `CLP.py` 直接呼叫 `LabRepository.retrieve_structured()`。

## 9. 啟動代理人前的必要條件

如果要使用 Lab KB，至少要確認：

1. 已把內部資料放到 `lab_kb/raw/`。
2. 已執行一次 `python lab_kb/build_lab_repo.py --rebuild`。
3. `lab_kb/store/` 中已產生 `vectors.npy`、`meta.json`，以及使用 FAISS 時的 `faiss.index`。
4. Python 環境已安裝 `sentence-transformers`、`numpy`、`scikit-learn`、`pypdf`；若使用 FAISS，需有 `faiss-cpu`。

之後即可正常啟動：

```bash
python CLP.py --cli --ollama_model <your-ollama-model>
```

## 10. 公開 GitHub 時不要提交的內容

請不要提交：

```text
lab_kb/raw/*
lab_kb/store/*
```

專案 `.gitignore` 已設定忽略這些內容，但保留兩個資料夾內的 `.gitkeep`。

如果你確定某些範例資料具有公開授權，可以另外建立例如：

```text
lab_kb/examples/
```

放置專門供重現測試使用的公開資料，而不要直接解除整個 `raw/` 的忽略設定。
