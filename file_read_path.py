import os
import pandas as pd
import glob
from dotenv import load_dotenv

def extract_project_id(file_path):
    # Extract the project ID from the file name
    filename = os.path.basename(file_path)
    project_id = filename.split("_")[0]
    return project_id

def read_csv_file(user_id, project_id):
    # Construct the directory path
    directory_path = f"/home/ubuntu/Satya/DataTango/tmp/{user_id}/gold/"

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    # Iterate over each CSV file
    for file_path in csv_files:
        # Extract the project ID from the file name
        file_project_id = extract_project_id(os.path.basename(file_path))

        # Check if the project ID matches the provided project_id
        if file_project_id == project_id:
            # Read the CSV file into a Pandas DataFrame
            print(file_path)
            df = pd.read_csv(file_path)

            return df
    else:
        print("Error: More than one CSV file found or no CSV file found in the directory.")
        return None

# Example usage
user_id = "65d5e841af6e64f8c032e8ed"
project_id = "6605470be7c1463e26b8fe29"  # Replace with the actual project_id
df = read_csv_file(user_id, project_id)
print(df.head(1))
