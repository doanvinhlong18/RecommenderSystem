"""
Script để test và verify dataset đã được download.
Sử dụng DataLoader từ preprocessing module.
"""
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import DATASET_PATH
from preprocessing import DataLoader

def main():
    print("=" * 60)
    print("ANIME RECOMMENDATION SYSTEM - Dataset Verification")
    print("=" * 60)

    # Verify dataset path
    print(f"\nDataset path: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset path does not exist!")
        print("Please ensure dataset is downloaded to the correct location.")
        return

    # List all files in dataset
    all_files = os.listdir(DATASET_PATH)
    print(f"\nAll files in dataset: {all_files}")

    # List CSV files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    print(f"CSV files: {csv_files}")

    # Quick load of all CSV files to verify
    print("\n" + "=" * 60)
    print("Loading datasets directly for verification...")
    print("=" * 60)

    datasets = {}
    for csv_file in csv_files:
        file_path = DATASET_PATH / csv_file
        dataset_name = csv_file.replace('.csv', '')
        try:
            # Load first few rows to check structure
            df = pd.read_csv(file_path, nrows=5)
            # Then count total rows
            total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1
            datasets[dataset_name] = {
                'sample': df,
                'rows': total_rows,
                'columns': len(df.columns)
            }
            print(f"✓ {csv_file}: {total_rows:,} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"✗ {csv_file}: Error - {e}")

    # Show sample of each dataset
    print("\n" + "=" * 60)
    print("Dataset Previews")
    print("=" * 60)

    for name, info in datasets.items():
        print(f"\n{name}:")
        print(f"  Columns: {list(info['sample'].columns)}")
        print(f"  First row: {info['sample'].iloc[0].to_dict()}")

    # Test DataLoader
    print("\n" + "=" * 60)
    print("Testing DataLoader module...")
    print("=" * 60)

    try:
        loader = DataLoader()

        # Load anime data
        anime_df = loader.load_anime()
        print(f"✓ Anime data loaded: {len(anime_df):,} records")
        print(f"  Columns: {list(anime_df.columns)[:10]}...")

        # Load synopsis
        synopsis_df = loader.load_synopsis()
        print(f"✓ Synopsis data loaded: {len(synopsis_df):,} records")

        # Get merged data
        merged_df = loader.get_merged_anime_data()
        print(f"✓ Merged anime data: {len(merged_df):,} records")

        # Load ratings (sampled)
        ratings_df = loader.load_ratings(sample=True)
        print(f"✓ Ratings data loaded: {len(ratings_df):,} records (sampled)")

        # Load watching status
        status_df = loader.load_watching_status()
        print(f"✓ Watching status loaded: {len(status_df):,} records")

        print("\n✓ All datasets loaded successfully!")
        print("\nYou can now run: python train.py")

    except Exception as e:
        print(f"✗ DataLoader error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
