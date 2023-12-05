import os

def rename_images(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Sort files by their creation time (modification time)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    # Iterate through the files and rename them
    for idx, file_name in enumerate(files, start=1):
        # Extract file extension
        _, extension = os.path.splitext(file_name)

        # Define the new file name with leading zeros and .jpg extension
        new_name = f"{idx:05d}.jpg"

        # Construct paths to the original and new file names
        current_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(current_path, new_path)
        print(f"Renamed {file_name} to {new_name}")

# Replace 'folder_path' with the path to your folder containing the images
folder_path = '/home/spike_03/SiamMask/data/traffic'
rename_images(folder_path)
