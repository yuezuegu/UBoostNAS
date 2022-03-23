
import json 
from io import StringIO

import src.hardware.pythonbinder as pythonbinder

def run_csim(json_out):
	io = StringIO()
	json.dump(json_out, io)
	json_dump = io.getvalue()

	no_array = 1
	no_rows = 128
	no_cols = 128
	bank_size = 5 * (1 << 20) #5 MB
	bandwidth = 80 #GB
	prefetch_limit = 100 
	interconnect_type = "crossbar"

	sim_res = pythonbinder.csim(json_dump, no_array, no_rows, no_cols, bank_size, bandwidth, prefetch_limit, interconnect_type)
	sim_res = json.loads(sim_res)

	return sim_res