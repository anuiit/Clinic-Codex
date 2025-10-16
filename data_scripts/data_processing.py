"""
Data processing utilities for the Codex project.
Generates CSV metadata files from the image dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import os


def create_image_dataframe(base_path, folders=['Elements', 'Glyphs']):
    """
    Create a comprehensive dataframe from images in Elements and Glyphs folders
    
    Args:
        base_path (str or Path): Base path to the project directory
        folders (list): List of folder names to process
    
    Returns:
        pandas.DataFrame: Dataframe with image information
    """
    data = []
    base_path = Path(base_path)
    
    for folder_type in folders:
        folder_path = base_path / folder_type
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist, skipping...")
            continue
            
        print(f"Processing {folder_type} folder...")
        processed_count = 0
        
        # Iterate through all subdirectories in the folder
        for category_dir in folder_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                
                # Extract the category code and name
                if '-' in category_name:
                    code, name = category_name.split('-', 1)
                else:
                    code, name = category_name, category_name
                
                # Process all images in this category
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                
                for image_file in category_dir.iterdir():
                    if image_file.suffix.lower() in image_extensions:
                        # Get image properties
                        try:
                            with Image.open(image_file) as img:
                                width, height = img.size
                                mode = img.mode
                                
                            # Calculate file size
                            file_size = image_file.stat().st_size
                            
                            # Create relative path from base directory
                            relative_path = image_file.relative_to(base_path)
                            
                            data.append({
                                'file_path': str(relative_path).replace('\\', '/'),  # Use forward slashes for cross-platform compatibility
                                'file_name': image_file.name,
                                'folder_type': folder_type,
                                'nb_of_appearances': code,
                                'word_name': name,
                                'full_category': category_name,
                                'width': width,
                                'height': height,
                                'image_mode': mode,
                                'file_size_bytes': file_size,
                                'file_extension': image_file.suffix.lower()
                            })
                            
                            processed_count += 1
                            if processed_count % 1000 == 0:
                                print(f"  Processed {processed_count} images...")
                                
                        except Exception as e:
                            print(f"Error processing {image_file}: {e}")
        
        print(f"✓ Completed {folder_type}: {processed_count} images processed")
    
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        # Add derived columns
        df['aspect_ratio'] = df['width'] / df['height']
        df['total_pixels'] = df['width'] * df['height']
        df['file_size_kb'] = df['file_size_bytes'] / 1024
        
        # Create a unique label combining folder type and category
        df['class_label'] = df['folder_type'] + '_' + df['full_category']
        
        # Sort by folder type and category
        df = df.sort_values(['folder_type', 'nb_of_appearances', 'file_name']).reset_index(drop=True)
    
    return df


def generate_csv_files(base_path):
    """
    Generate all CSV metadata files for the dataset.
    
    Args:
        base_path (str or Path): Base path to the project directory
        
    Returns:
        dict: Dictionary with information about generated files
    """
    base_path = Path(base_path)
    print("🔍 Analyzing dataset and generating CSV files...")
    
    # Create the main dataframe
    df_images = create_image_dataframe(base_path)
    
    if len(df_images) == 0:
        print("❌ No images found in dataset!")
        return {"success": False, "message": "No images found"}
    
    print(f"📊 Found {len(df_images)} total images")
    
    # Generate different CSV variants
    results = {}
    
    try:
        # 1. Complete dataframe with all columns
        all_data_path = base_path / "image_data_alljpg.csv"
        df_images.to_csv(all_data_path, index=False)
        results["all_data"] = {"path": all_data_path, "rows": len(df_images)}
        print(f"✓ Generated {all_data_path.name} ({len(df_images)} rows)")
        
        # 2. Summary version (without file size columns)
        df_summary = df_images.drop(columns=['file_extension', 'file_size_bytes', 'file_size_kb'], errors='ignore')
        summary_path = base_path / "image_data_summary_v2.csv"
        df_summary.to_csv(summary_path, index=False)
        results["summary"] = {"path": summary_path, "rows": len(df_summary)}
        print(f"✓ Generated {summary_path.name} ({len(df_summary)} rows)")
        
        # 3. Legacy summary format (for backwards compatibility)
        legacy_path = base_path / "image_data_summary.csv" 
        df_images.to_csv(legacy_path, index=False)
        results["legacy"] = {"path": legacy_path, "rows": len(df_images)}
        print(f"✓ Generated {legacy_path.name} ({len(df_images)} rows)")
        
        # Print statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total images: {len(df_images):,}")
        
        if 'folder_type' in df_images.columns:
            folder_stats = df_images['folder_type'].value_counts()
            for folder, count in folder_stats.items():
                print(f"   {folder}: {count:,} images")
        
        if 'full_category' in df_images.columns:
            unique_categories = df_images['full_category'].nunique()
            print(f"   Unique categories: {unique_categories}")
            
        print(f"   Image formats: {df_images['file_extension'].value_counts().to_dict()}")
        
        results["success"] = True
        results["total_images"] = len(df_images)
        
    except Exception as e:
        print(f"❌ Error generating CSV files: {e}")
        results["success"] = False
        results["error"] = str(e)
    
    return results


def main():
    """CLI entry point for generating CSV files."""
    import sys
    from pathlib import Path
    
    # Get the base path (directory containing this script)
    base_path = Path(__file__).parent
    
    print("🚀 Starting CSV generation...")
    results = generate_csv_files(base_path)
    
    if results["success"]:
        print(f"\n✅ CSV generation completed successfully!")
        print(f"Generated {len([k for k in results.keys() if k != 'success'])} CSV files")
    else:
        print(f"\n❌ CSV generation failed!")
        if "error" in results:
            print(f"Error: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()