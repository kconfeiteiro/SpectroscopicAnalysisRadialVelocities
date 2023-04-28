from Package import AnalyzeSpectra, ImportSpectra, PlotSpectra
from typing import Tuple, Dict, List
import datetime as dt
import pandas as pd
import os, re

'''
This file will be responsible 
'''

now = dt.datetime.now()
today_time = now.strftime('%Y_%m_%d_%H%M%S')
today = now.strftime('%Y_%m_%d')

class Report(ImportSpectra):

	def __init__(self, path:str, save_to=f'{today}/REPORT/'):
		super().__init__(path)
		self.save_to = save_to

	def summary(self):
		print(f'''
			FILE: \'{self.filename}\'
			PATH: \'{self.path}\'
			TARGET: \'{self.target}\'
			NUM. FILES: \'{self.num_files}\'
		''')

	def write_to_text(self, filename: str = None, lines: List or str = None):

		with open(f'{self.save_to}{filename}', 'w') as file:
			if type(lines) == str:
				file.write(lines)
			elif type(lines) == list:
				for line in lines:
					line = str(line)
					file.write(line + '\n')

		return print(f'Your file has been saved as \'{filename}\'')