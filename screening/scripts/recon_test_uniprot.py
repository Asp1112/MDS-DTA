import requests

q = '(ec:2.3.1.181 OR protein_name:"octanoyltransferase" OR gene:lipb) AND fragment:false'
r = requests.get(
    "https://rest.uniprot.org/uniprotkb/search",
    params={"query": q, "format": "json", "size": "5"},
    timeout=60,
)
print("status", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("total results:", len(data.get("results", [])))
    for rec in data.get("results", [])[:3]:
        print(rec.get("primaryAccession"), rec.get("organism", {}).get("scientificName", ""))
