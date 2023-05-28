import pdfplumber
from pymongo import MongoClient
import sys

'''
How to run:
$ python pdf_extractor .py PACIENT_ID PDF_FILE
'''

# Arguments
pacient_cnp = sys.argv[1] # pacient_id here 645430a43b0ec4b7df36aec6
pdf_file = sys.argv[2]

def split_subfields(value):
    subfields = {}
    # Special cases for the lines starting with "Bilirubina totala" and "Inaltime (cm)"
    if value.startswith("Bilirubina totala") or value.startswith("Inaltime (cm)"):
        # Replace ". ," with a temporary string to prevent incorrect splitting
        temp_value = value.replace(". ,", "<TEMP>")
        parts = temp_value.split(", ")
        for part in parts:
            part = part.replace("<TEMP>", ". ,")  # replace the temporary string back to ". ,"
            if "=" in part:
                sf_field, sf_value = part.split('=', 1)
                subfields[sf_field.strip()] = sf_value.strip()
            else:
                subfields[part] = ''
        return subfields
    # Case for "Hemoleucograma completa"
    if value.startswith("["):
        value = value[1:-1]  # remove square brackets
        parts = value.split(", ")
        for part in parts:
            if "=" in part:
                sf_field, sf_value = part.split('=', 1)
                subfields[sf_field.strip()] = sf_value.strip()
            else:
                subfields[part] = ''
        return subfields
    # Regular case
    if ',' in value and '=' in value:
        parts = value.split(', ')
        for part in parts:
            if '=' in part:
                sf_field, sf_value = part.split('=', 1)
                sf_field = sf_field.strip()
                sf_value = sf_value.strip()
                subfields[sf_field] = sf_value
            else:
                subfields[part] = ''
        return subfields
    else:
        return value

# Load PDF
with pdfplumber.open(pdf_file) as pdf:
    first_page = pdf.pages[0]
    text = first_page.extract_text()

# Find start of section
start_index = text.find('BILET DE IESIRE / SCRISOARE MEDICALA')
section_text = text[start_index:]

# Split text into lines
lines = section_text.split('\n')

# Prepare dictionary
data_dict = {}
current_field = None
for line in lines:
    if ':' in line:  # Assuming that each field is separated by a ':'
        field, value = line.split(':', 1)  # Split at first occurrence of ':'
        field = field.strip()  # Remove leading/trailing whitespace
        value = value.strip()  # Remove leading/trailing whitespace
        current_field = field
        data_dict[field] = split_subfields(value)
    elif current_field is not None:
        # This line is a continuation of the last field
        subfields = split_subfields(line)
        if isinstance(subfields, dict):
            # Merge the subfields into the current field
            if isinstance(data_dict[current_field], dict):
                data_dict[current_field].update(subfields)
            else:
                data_dict[current_field] = subfields
        else:
            # Add this line as a new value in the current field
            if isinstance(data_dict[current_field], list):
                data_dict[current_field].append(subfields)
            else:
                data_dict[current_field] = [data_dict[current_field], subfields]

# Convert the "Hemoleucograma completa" dictionary to a BSON object
hemoleucograma_completa_dict = data_dict.get("Hemoleucograma completa", {})

buletin_key = None
for key in data_dict:
    if "Buletin" in key:
        buletin_key = key
        break

buletin_date = buletin_key.split(" ")[1]
hemoleucograma_completa_dict["date"] = buletin_date

# Open mongo
MONGO_CONNECTION_STRING = "mongodb+srv://dianavelciov:parola@cluster0.qqmezlq.mongodb.net/cool_notes_app?retryWrites=true&w=majority"

# Create a MongoClient to the running MongoDB Atlas cluster instance
client = MongoClient(MONGO_CONNECTION_STRING)

# Getting a Database
db = client.cool_notes_app

# Getting a Collection
collection = db.patients

result = collection.update_one({"cnp": pacient_cnp}, {"$push": {"Hemoleucograma_completa": hemoleucograma_completa_dict}}, upsert=True)
