import os
import json
import enum
from time import monotonic
from typing import Optional

import requests
from fastapi import FastAPI, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .functions import (
    define_peptide_generated_descriptors,
    calculate_monomer
)


app = FastAPI()

@app.get("/peptide_descriptors")
def get_peptide_descriptors(
        sequence: str = Query(default="")
):
    return define_peptide_generated_descriptors(sequence)

@app.get("/calculated_monomer")
def get_calculated_monomer(
        designation: str,
        smiles: str,
        polymer_type: str
):
    return calculate_monomer(designation, smiles, polymer_type)