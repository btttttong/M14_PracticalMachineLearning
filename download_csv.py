import os
import pandas as pd

# URL of the CSV file
csv_url = "https://raw.githubusercontent.com/aaaksenova/harbour_space/change/train_cat_breeds.csv"
# Path to save the downloaded CSV
metadata_csv_path = "./data/original_metadata.csv"
# Directory containing breed subdirectories
image_root_dir = "./data/images"
# Output CSV path
output_csv_path = "./data/train_cat_breeds.csv"

# Download the original metadata CSV
print("Downloading the CSV file...")
df = pd.read_csv(csv_url)
df.to_csv(metadata_csv_path, index=False)
print(f"CSV file downloaded and saved at {metadata_csv_path}")

# Load the original metadata CSV
metadata_df = pd.read_csv(metadata_csv_path)

# Add image paths based on the 'id' column
metadata_df["image_path"] = metadata_df.apply(
    lambda row: os.path.normpath(
        os.path.join(image_root_dir, row["breed"], f"{row['id']}.jpg")
    ),
    axis=1,
)

# Debugging: Print the first few rows to check the paths
print("First few image paths:")
print(metadata_df[["id", "breed", "image_path"]].head())

# Check if the images exist and filter out the rows where the image doesn't exist
metadata_df["image_exists"] = metadata_df["image_path"].apply(os.path.exists)

# Debugging: Print number of images found
print(f"Total images found: {metadata_df['image_exists'].sum()}")

# Filter out rows without images
metadata_df = metadata_df[metadata_df["image_exists"]]

# Debugging: Print the filtered DataFrame
print("Filtered DataFrame:")
print(metadata_df.head())

# Drop the helper column
metadata_df = metadata_df.drop(columns=["image_exists"])

# Save to CSV
metadata_df.to_csv(output_csv_path, index=False)

# Debugging: Confirm saving of CSV
print(f"CSV file created at {output_csv_path}")

# Debugging: Print final CSV contents
print("Final CSV contents:")
print(metadata_df.head())
