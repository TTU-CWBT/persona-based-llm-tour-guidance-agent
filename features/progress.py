# features/progress.py
import os


def configure_progress(enabled: bool) -> None:
    """設定與進度列相關的環境變數。

    在目前 Ollama-only 的架構下，我們不再直接操作 transformers / datasets 的進度列，
    只簡單透過環境變數控制 TQDM 顯示即可。
    """
    # 控制 TQDM 是否顯示
    os.environ["TQDM_DISABLE"] = "0" if enabled else "1"

    # 若之後你還是有用到 HF 相關工具，這兩行保留也無妨；
    # 就算沒用，留著也只是一個無害的 env 設定。
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0" if enabled else "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 目前不再 import transformers / datasets 來動態調整 progress bar，
    # 避免在這個專案中對 HF 產生實際依賴。
    return
