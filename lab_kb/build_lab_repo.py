from __future__ import annotations
import os, yaml, argparse
from lab_repository import LabRepository, LabRepoConfig


def _collect_new_raw_paths(repo: LabRepository) -> list[str]:
    raw_paths = repo.iter_raw_files()
    indexed_paths = repo.indexed_source_paths()
    return [p for p in raw_paths if p not in indexed_paths]

def load_cfg(path: str) -> LabRepoConfig:
    config_path = os.path.abspath(path)
    config_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    def _resolve_dir(value: str) -> str:
        value = str(value)
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(config_dir, value))

    return LabRepoConfig(
        raw_dir=_resolve_dir(y["raw_dir"]),
        store_dir=_resolve_dir(y["store_dir"]),
        embedding_model=y.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
        normalize_embeddings=y.get("normalize_embeddings", True),
        chunk_size=y.get("chunk_size", 800),
        chunk_overlap=y.get("chunk_overlap", 120),
        top_k=y.get("top_k", 4),
        min_score=y.get("min_score", 0.30),
        include_ext=tuple(y.get("include_ext", [".txt", ".md", ".pdf", ".csv"])),
        use_faiss=y.get("use_faiss", True),
        strip_whitespace=y.get("strip_whitespace", True),
        collapse_newlines=y.get("collapse_newlines", True),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "lab_kb_config.yaml"),
        help="path to yaml",
    )
    ap.add_argument("--rebuild", action="store_true", help="rebuild from scratch")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    repo = LabRepository(cfg)

    if args.rebuild:
        print("Rebuilding lab repository from raw files...")
        repo.build()
        print("Done.")
    else:
        print("Incremental add from raw files (if new) ...")
        new_paths = _collect_new_raw_paths(repo)
        if not new_paths:
            print("No new raw files found. Repository is up to date.")
        else:
            print(f"Found {len(new_paths)} new file(s). Adding to repository...")
            repo.add_paths(new_paths)
            print("Done.")

if __name__ == "__main__":
    main()
