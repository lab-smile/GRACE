from functions import *


#list of file names
FILES = []

if __name__ == "__main__":
    if len(FILES) == 0:
        print("Please add files to FILES list.")
        quit()
    
    complete = convert_to_nii(FILES)
    if complete:
        print("Completed")