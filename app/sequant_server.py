import os
import json
import enum
from time import monotonic
from typing import Optional

import requests
from fastapi import FastAPI, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .functions import define_peptide_generated_descriptors


app = FastAPI()

@app.get("/peptide_descriptors")
def get_peptide_descriptors(
    sequences: list = Query(default=[])
):
    return json.loads(define_peptide_generated_descriptors(sequences))
