import sys
import os

DATA_FILES = 'output/'
for file in os.listdir(DATA_FILES):
	if file.split('.')[1] == 'txt':
		content = ''
		with open(DATA_FILES + file, 'r') as in_f:
			content = in_f.read()
			content = content.replace('\n', '')
			content = content.replace('\r', '')
			content = content.replace('\t', '')
			content = unicode(content, 'ascii', 'ignore')
		with open(DATA_FILES + file, 'w') as out_f:
			out_f.write(content)

