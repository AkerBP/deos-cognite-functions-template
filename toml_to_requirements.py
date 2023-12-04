import os
import re
import shutil

def get_toml_dependencies():
    toml = "pyproject.toml"
    with open(toml, "r") as file:
        content = file.read()

        pattern = r"\[tool.poetry.dependencies\](.*?)\["
        match = re.findall(pattern, content, re.DOTALL)[0]
        match = match.replace('"', '')
        # match = re.sub(r"= (?=[0-9])", "== ", match)
        match = re.sub(r"= [0-9\.]+", "", match)
        # match = re.sub(r"= \^(?=[0-9])", ">= ", match)
        match = re.sub(r"= \^[0-9\.]+", "", match)
        match = re.sub(r"python(?![^ ])", "", match)

        with open("requirements.txt", "w") as out_file:
            print(f"Created requirements.txt in {os.getcwd()}")
            out_file.write(match)
            out_file.close()

        file.close()

def move_file_to_subfolders(file_path, subfolder_prefix):
    src_directory = os.getcwd() + "\src"
    subfolders = [folder for folder in os.listdir(src_directory) if os.path.isdir("src/"+folder) and folder.startswith(subfolder_prefix)]

    for subfolder in subfolders:
        destination_path = os.path.join(src_directory, subfolder, os.path.basename(file_path))
        shutil.copy(file_path, destination_path)
        print(f"Moved {file_path} to {destination_path}")
    print(f"Removed {file_path} from {os.getcwd()}")
    os.remove(file_path)

if __name__ == "__main__":
    get_toml_dependencies()

    file_to_move = "requirements.txt"
    subfolder_prefix = "cf_"

    move_file_to_subfolders(file_to_move, subfolder_prefix)

