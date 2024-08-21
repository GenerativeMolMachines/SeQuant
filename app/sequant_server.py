import os
import json
import enum
from time import monotonic
from typing import Optional

import requests
from fastapi import FastAPI, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from sequant_main import SeqQuantKernel


app = FastAPI()

@app.get("/encode_sequence")
def get_peptide_descriptors(
        sequences: str = Query(default=""),
        polymer_type: str = 'protein',
        encoding_strategy: str ='protein',
        new_monomers: str = '',
        skip_unprocessable: bool = True
):
    print(sequences)
    print(new_monomers)
    if new_monomers != '':
        new_monomers_dict = json.loads(new_monomers)
    else:
        new_monomers_dict = {}
    print(new_monomers)
    sequence_list = sequences.split(",")
    sqk = SeqQuantKernel(
        polymer_type=polymer_type,
        encoding_strategy=encoding_strategy,
        new_monomers=new_monomers_dict,
    )
    return sqk.generate_latent_representations(sequence_list, skip_unprocessable)