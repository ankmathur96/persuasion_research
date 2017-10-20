import os
import sys
import requests
from bs4 import BeautifulSoup

BASE_URL = 'http://www.americanrhetoric.com'
SPEECH_BANK = '/top100speechesall.html'
# http://www.americanrhetoric.com/top100speechesall.html
OUTPUT_DIR = 'output/'

def get_speech_links(html_text):
	parsed_str = BeautifulSoup(html_text, 'html.parser')
	content_level = parsed_str.find_all('a')
	filtered = [x.get('href') for x in content_level if '.htm' in x.get('href') and 'speeches/' in x.get('href')]
	return filtered

def get_speech(html_text):
	parsed_str = BeautifulSoup(html_text, 'html.parser')
	content_level = parsed_str.find_all('p')
	filtered = [x.font.get_text() for x in content_level if x.get('align') == 'left' and x.font is not None]
	speech_text = " ".join(filtered)
	return speech_text

r = requests.get(BASE_URL + SPEECH_BANK)
all_speeches = get_speech_links(r.text)
print('Identified ' + str(len(all_speeches)) + ' speeches to fetch.')
for i, website in enumerate(all_speeches):	
	speech_name = website.split('/')[-1].split('.')[0]
	print('fetching speech id ' + str(i) + ', named ' + str(speech_name))
	request_url = BASE_URL + '/' + website
	r = requests.get(request_url)
	speech = get_speech(r.text)
	with open(OUTPUT_DIR + speech_name + '.txt', 'w') as out_f:
		out_f.write(speech)
	print('wrote speech id ' + str(i) + ', named ' + str(speech_name))


# TEST_URL = 'http://www.americanrhetoric.com/speeches/convention2004/barackobama2004dnc.htm'
# r = requests.get(TEST_URL)
# print(get_speech(r.text))

