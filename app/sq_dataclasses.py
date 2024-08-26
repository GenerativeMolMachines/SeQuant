from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class PolymerType(str, Enum):
    protein = 'protein'
    dna = 'DNA'
    rna = 'RNA'


class EncodingStrategy(str, Enum):
    protein = 'protein'
    aptamers = 'aptamers'


class Monomer(BaseModel):
    name: str = Field(max_length=1)
    smiles: str


class NewMonomers(BaseModel):
    monomers: List[Monomer]
