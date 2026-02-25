import os
import pandas as pd
import numpy as np

def diagnose():
    labels_file = 'labels.txt'
    img_dir = 'images'
    output_file = 'diagnose_results.txt'
    
    results = []
    
    if not os.path.exists(labels_file):
        results.append(f"Error: {labels_file} not found.")
    else:
        df = pd.read_csv(labels_file, sep=' ', header=None)
        df.columns = ['img_name', 'attr1', 'attr2', 'attr3', 'attr4']
        
        results.append(f"Total entries in labels.txt: {len(df)}")
        
        missing_images = []
        for img_name in df['img_name']:
            if not os.path.exists(os.path.join(img_dir, img_name)):
                missing_images.append(img_name)
                
        results.append(f"Number of missing images: {len(missing_images)}")
        if missing_images:
            results.append(f"First 5 missing: {', '.join(missing_images[:5])}")
            
        # Check for NA and types
        results.append("\nLabel distribution (including NA):")
        for col in df.columns[1:]:
            results.append(f"{col}:")
            counts = df[col].value_counts(dropna=False).to_string()
            results.append(counts)
            
        # Check for actual NaN (not 'NA' strings)
        results.append("\nActual NaN counts in DataFrame:")
        results.append(df.isna().sum().to_string())

    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"Results written to {output_file}")

if __name__ == '__main__':
    diagnose()
