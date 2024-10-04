from fastapi import FastAPI, Query

from sequant_main import SeqQuantKernel
from sq_dataclasses import (
    PolymerType,
    EncodingStrategy,
    NewMonomers,
    SeqQuantKernelModel
)


app = FastAPI()

@app.post("/encode_sequence")
def get_peptide_descriptors(
        sequences: str = Query(default=""),
        polymer_type: PolymerType = 'protein',
        encoding_strategy: EncodingStrategy ='protein',
        new_monomers: NewMonomers = None,
        skip_unprocessable: bool = True
):
    sequence_list = sequences.replace(" ", "").split(",")
    sqk = SeqQuantKernel(
        polymer_type=polymer_type,
        new_monomers=new_monomers
    )
    return sqk.generate_latent_representations(
        sequence_list=sequence_list,
        skip_unprocessable=skip_unprocessable,
        encoding_strategy=encoding_strategy
    )


@app.get("/monomers/{polymer_type}")
def get_existing_monomers(polymer_type: PolymerType = 'protein'):
    sqk = SeqQuantKernel(
        polymer_type=polymer_type
    )
    return sqk.known_monomers


@app.post("/kernel_info")
def get_kernel_info(
        polymer_type: PolymerType = 'protein',
        new_monomers: NewMonomers = None
    ):
    sqk = SeqQuantKernel(
        polymer_type=polymer_type,
        new_monomers=new_monomers
    )
    return SeqQuantKernelModel(
        max_sequence_length=sqk.max_sequence_length,
        num_of_descriptors=sqk.num_of_descriptors,
        known_monomers=sqk.known_monomers,
        polymer_type=sqk.polymer_type
    )