"""
Helper file to decompress forward-facing 
"""

import os

face_dir = '../data/colorferet100'
people = os.listdir(face_dir)

for p in people:
	p_dir = os.path.join(face_dir,p)

	try:
		p_files = os.listdir(p_dir)
	except NotADirectoryError:
		print('This is not a directory: ' + p_dir)
		continue


	for f in p_files:
		if 'fa' in f:
			to_unzip = os.path.join(p_dir,f)
			os.system('sudo bzip2 -d ' + to_unzip)
			print(to_unzip)
