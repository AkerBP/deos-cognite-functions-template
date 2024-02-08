import argparse
import os
import shutil

def create_cognite_function_project(project_name):
    # Define the source and destination paths
    template_dir = os.path.join(os.path.dirname(__file__), 'standard_template')
    destination_dir = os.path.join(os.getcwd(), project_name)

    # Copy the files from the template directory to the new project directory
    shutil.copytree(template_dir, destination_dir)
    print(f'New Cognite Function project "{project_name}" has been created from template.')

def main():
    parser = argparse.ArgumentParser(description="cognite_functions_template CLI")
    parser.add_argument('new_cognite_function', type=str, help='Name of the new Cognite Function')
    args = parser.parse_args()

    create_cognite_function_project(args.new_project)

if __name__ == "__main__":
    main()
