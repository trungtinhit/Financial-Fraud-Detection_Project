from pathlib import Path
import shutil
import sys

# ============================================================
# IMPORT PROJECT ROOT
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CURRENT_DIR / "config"

if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from paths import PROJECT_ROOT


# ============================================================
# CONFIG
# ============================================================

CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

CACHE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_cache():
    removed_dirs = 0
    removed_files = 0

    print("=" * 70)
    print("CLEAR PYTHON CACHE")
    print("=" * 70)

    print(f"\nPROJECT_ROOT:\n{PROJECT_ROOT}\n")

    # --------------------------------------------------------
    # XÓA CACHE FOLDERS
    # --------------------------------------------------------

    for path in list(PROJECT_ROOT.rglob("*")):
        if path.is_dir() and path.name in CACHE_DIR_NAMES:
            try:
                shutil.rmtree(path)
                print(f"[DIR]  Removed: {path}")
                removed_dirs += 1
            except Exception as e:
                print(
                    f"[ERROR] Cannot remove directory: "
                    f"{path}\n        {e}"
                )

    # --------------------------------------------------------
    # XÓA CACHE FILES
    # --------------------------------------------------------

    for path in list(PROJECT_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in CACHE_FILE_SUFFIXES:
            try:
                path.unlink()
                print(f"[FILE] Removed: {path}")
                removed_files += 1
            except Exception as e:
                print(
                    f"[ERROR] Cannot remove file: "
                    f"{path}\n        {e}"
                )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HOÀN THÀNH")
    print("=" * 70)

    print(f"Cache folders removed: {removed_dirs}")
    print(f"Cache files removed:   {removed_files}")

    if removed_dirs == 0 and removed_files == 0:
        print("\nKhông tìm thấy cache cần xóa.")
    else:
        print("\nĐã dọn cache Python thành công.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    clear_cache()
