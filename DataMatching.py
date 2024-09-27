import pandas as pd

# Reading Excel files
df_a = pd.read_excel('FilePath') # Well testing conclusion file
df_b = pd.read_excel('FilePath') # Depth points to match file

# Create a new column to store the oil test results
df_b['Well testing conclusion'] = None

# Traverse table B
for i, row_b in df_b.iterrows():
    well_name_b = row_b['Well names']
    depth_b = row_b['Depth']

    # Filter out rows with the same well name in table A
    relevant_a_rows = df_a[df_a['Well names'] == well_name_b]

    for j, row_a in relevant_a_rows.iterrows():
        top_depth_a = row_a['Top perforation depth']
        bottom_depth_a = row_a['Perforation depth']
        conclusion_a = row_a['Well testing conclusion']

        # Determine whether the depth in table B is within the depth range of table A
        if top_depth_a <= depth_b <= bottom_depth_a:
            df_b.at[i, 'Well testing conclusion'] = conclusion_a
            break  # If we find a match, we can get out of the inner loop

# Save the results to a new Excel file
df_b.to_excel('FilePath', index=False)

'''Screening of non-air test results'''
import os
import time
import pandas as pd
from multiprocessing import Pool

# Source folder path and destination folder path
source_folder = 'FilePath' #Gas logging data (after matching oil test results）
target_folder = 'FilePath'

# Make sure the destination folder exists
if not os.path.exists(target_folder):
    os.makedirs(target_folder)


# Define a function to process a single Excel file
def process_excel(file_path):
    engine = None
    if file_path.endswith('.xlsx'):
        engine = 'openpyxl'  # Explicitly specify that the openpyxl engine handles.xlsx files
    elif file_path.endswith('.xls'):
        try:
            import xlrd  # Try to import the xlrd library
            engine = 'xlrd'  # If xlrd is available, use it to process.xls files
        except ImportError:
            print(f"xlrd not installed, skipping {file_path} (openpyxl does not support .xls).")
            return  # Skip the processing of.xls files because openpyxl does not support them

    try:
        df = pd.read_excel(file_path, engine=engine)
        filtered_df = df[df['Well testing conclusion'].notnull()]

        # Construct the destination file path and file name (assuming that the source and destination folders are in the same parent directory)
        base_name = os.path.basename(file_path)
        target_file_path = os.path.join(target_folder, base_name)

        # If the filtered data is not empty, a new Excel file is written
        if not filtered_df.empty:
            filtered_df.to_excel(target_file_path, index=False)
            print(f"Processed {base_name} and saved to {target_file_path}")
        else:
            print(f"No non-empty rows in 'Well testing conclusion' column in {base_name}, skipped.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


start_time = time.time()

# Gets the path to all Excel files in the source folder
excel_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if
               f.endswith('.xlsx') or f.endswith('.xls')]

if __name__ == '__main__':
    with Pool() as pool:  # By default, the number of CPU cores is used as the number of processes
        pool.map(process_excel, excel_files)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing completed.")
    print(f"Running time: {elapsed_time} 秒")