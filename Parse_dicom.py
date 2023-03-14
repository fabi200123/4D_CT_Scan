import os
import pydicom

dir_path = "fabi_data/"

# Function to parse the DICOM images
def Parse_dicom(dir_path):
    num_scans = len(os.listdir(dir_path))
    slices_list = []
    print(num_scans)
    i = 0

    for i in range(num_scans):
        # load DICOM files
        files = []
        dicom_dir = dir_path + "scan_" + str(i) + "/"
        # Loop through all files in the directory
        for filename in os.listdir(dicom_dir):
            # Check if the file is a DICOM file
            if filename.endswith(".dcm"):
                # Read the DICOM file and append it to the list
                file_path = os.path.join(dicom_dir, filename)
                files.append(pydicom.dcmread(file_path))
        print("file count: {}".format(len(files)))

        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, 'SliceLocation'):
                slices.append(f)
            else:
                skipcount = skipcount + 1

        print("skipped, no SliceLocation: {}".format(skipcount))

        # ensure they are in the correct order
        slices = sorted(slices, key=lambda s: s.SliceLocation)
        slices_list.append(slices)

    return slices_list

# Testing
slices_list = Parse_dicom(dir_path)
print(slices_list[0])