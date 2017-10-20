import os
import sys
import requests
from bs4 import BeautifulSoup
# configuration for whatthefolly. Configuration must be learned by user.
# domain that this configuration is for
BASE_DOMAIN = 'http://www.whatthefolly.com'
# tags that can safely be discarded to parse the speech text.
DISCARD_TAGS = ['<a', '<span', '<strong']
# <p> classes that shouldn't be included in the text.
BLACKLIST_P_CLASSES = ['nocomments']
# strings that the website uses to indicate pagination or multiple parts
CONTINUTATION_STRINGS = ['...', '…']
# strings that the website uses to indicate last page
TERMINATION_STRINGS = ['###']
# directory to write output to.
ROOT_DIR = 'speeches/'


def extract_text(html_text):
	parsed_str = BeautifulSoup(html_text, 'html.parser')
	content_level = parsed_str.body.div.find_all('p')
	filtered_speech = []
	for p_tag in content_level:
		discard = False
		if 'class' in p_tag.attrs and p_tag.attrs['class'][0] in BLACKLIST_P_CLASSES:
			discard = True
		elif p_tag.string and (p_tag.string in CONTINUTATION_STRINGS or p_tag.string in TERMINATION_STRINGS):
			discard = True
		else:
			for tag in DISCARD_TAGS:
				if tag in str(p_tag):
					discard = True
		if not discard:
			filtered_speech.append(p_tag.string)
	speech_content = ''
	for s in filtered_speech:
		if s is not None:
			speech_content += str(s) + '\n'
	return filter_html_tags(speech_content)

# works under assumption < has a > tag or the string end and relevant text does NOT contain '>' or '<'.
def filter_html_tags(s):
	filtered = ''
	open_brackets = 0
	for c in s:
		if c == '<':
			open_brackets += 1
		elif c == '>':
			if open_brackets > 0:
				open_brackets -= 1
				filtered += '\n'
		elif open_brackets <= 0:
			filtered += c
	return filtered

def test():
	r = requests.get('http://www.whatthefolly.com/2016/10/21/transcript-hillary-clintons-speech-in-cleveland-ohio-part-10/')
	extracted = extract_text(r.text)
	print(extracted)

if __name__ == '__main__':
	# test()
	try:
		base_url, n_parts = sys.argv[1], int(sys.argv[2])
	except Exception:
		base_url, n_parts = 'http://www.google.com', 0
	speech_name = base_url.split('/')[-1]
	link_completer = '-part-'
	all_text_in_speech = ''
	for i in range(1, n_parts + 1):
		print('making request for part ' + str(i))
		r = requests.get(base_url + link_completer + str(i))
		all_text = extract_text(r.text)
		all_text_in_speech += all_text + '\n'
	with open(ROOT_DIR + speech_name + '.txt', 'w') as speech_out:
		speech_out.write(all_text_in_speech)



