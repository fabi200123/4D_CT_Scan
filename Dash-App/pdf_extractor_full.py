# importing required modules
from PyPDF2 import PdfReader
import re
import json

# creating a pdf reader object
reader = PdfReader("bilet.pdf")

text = ""

# getting a specific page from the pdf file;
for i in range(len(reader.pages)):
    page = reader.pages[i]
    # extracting text from page
    text += page.extract_text()

data_dict = {}

# Extracting data using refined regular expressions
data_dict['Address'] = re.search(r"Str\.(.*?)\n", text).group(1).strip()
data_dict['Tel'] = re.search(r"Tel: (.*?);", text).group(1).strip()
data_dict['Fax'] = re.search(r"Fax: (.*?)\n", text).group(1).strip()
data_dict['e-mail'] = re.search(r"e-mail: (.*?)\n", text).group(1).strip()
data_dict['Nr. contract/conventie'] = re.search(r"Nr\. contract/conventie: (.*?)\n", text).group(1).strip()
data_dict['Nume'] = re.search(r"Nume:  (.*?)Domiciliu:", text).group(1).strip()
data_dict['Domiciliu'] = re.search(r"Domiciliu:     (.*?)Data internare:", text).group(1).strip()
data_dict['Data internare'] = re.search(r"Data internare:  (.*?)\s", text).group(1).strip()
data_dict['C.N.P'] = re.search(r"C\.N\.P: (\d+)", text).group(1).strip()
data_dict['Jud'] = re.search(r"Jud:  (\w+)", text).group(1).strip()
data_dict['Data externare'] = re.search(r"Data externare:  (.*?)\s", text).group(1).strip()
data_dict['Data nastere'] = re.search(r"Data nastere: (\d{2}/\d{2}/\d{4})", text).group(1).strip()
data_dict['Varsta'] = re.search(r"Varsta:  (\d+ ani)", text).group(1).strip()
data_dict['Sex'] = re.search(r"Sex:  (\w)", text).group(1).strip()
data_dict['Grup sangvin'] = re.search(r"Grup sangvin:  (\w) \|", text).group(1).strip()
data_dict['RH'] = re.search(r"RH:  (\w+)?", text).group(1).strip() if re.search(r"RH:  (\w+)?", text) else ""
data_dict['Alergii'] = re.search(r"Alergii:  (.*?)\n", text).group(1).strip()
data_dict['Diagnostic'] = re.search(r"Diagnostic: ((.|\n)*)(.*?)(?=TNM)", text).group(1).strip()
data_dict['TNM'] = re.search(r"TNM:(\w+)?", text).group(1).strip()
data_dict['Stadiul'] = re.search(r"Stadiul: (\w+)?", text).group(1).strip()
data_dict['Ex. histopatologic'] = re.search(r"Ex. histopatologic: ((.|\n)*)(.*?)(?=Investigatii)", text).group(1).strip()


json_object = json.dumps(data_dict, indent = 4) 
print(json_object)