import os
import json
import enum
from time import monotonic
from typing import Optional

import requests
from fastapi import FastAPI, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse


rs_server = FastAPI()

