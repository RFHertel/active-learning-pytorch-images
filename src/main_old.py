import sys

from data.notebooks.original_pipeline import run_original_pipeline
from data.notebooks.updated_pipeline import run_updated_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [original|updated]")
        print("Defaulting to 'updated'.")
        stream = 'updated'
    else:
        stream = sys.argv[1].lower()

    if stream == 'original':
        print("Running Original Pipeline...")
        run_original_pipeline()
    elif stream == 'updated':
        print("Running Updated Pipeline...")
        run_updated_pipeline()
    else:
        print("Invalid stream. Choose 'original' or 'updated'.")