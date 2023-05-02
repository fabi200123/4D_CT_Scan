import SimpleITK as sitk

# Replace 'path_to_your_nrrd_file' with the actual path to the NRRD file
nrrd_file = "C:\\Users\\fabi2\\OneDrive\\Documents\\GitHub\\4D_CT_Scan\\Visual\\data\\converted_nrrds\\LUNG1-001_20180209_CT_2\\GTV-1_mask.nrrd"

image = sitk.ReadImage(nrrd_file)
spacing = image.GetSpacing()

print(f"Spacing: {spacing}")