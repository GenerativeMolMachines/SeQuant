import os
import json
import enum
from time import monotonic
from typing import Optional

import requests
from fastapi import FastAPI, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from sequant_main import SeqQuantKernel
from sq_dataclasses import *


app = FastAPI()

@app.post("/encode_sequence")
def get_peptide_descriptors(
        sequences: str = Query(default=""),
        polymer_type: PolymerType = 'protein',
        encoding_strategy: EncodingStrategy ='protein',
        new_monomers: NewMonomers = None,
        skip_unprocessable: bool = True
):
    sequence_list = sequences.split(",")
    sqk = SeqQuantKernel(
        polymer_type=polymer_type,
        encoding_strategy=encoding_strategy,
        new_monomers=new_monomers,
    )
    return sqk.generate_latent_representations(sequence_list, skip_unprocessable)