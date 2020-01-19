#Author : Muhammad Azmi
import re
import string
import nltk.probability
# from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
# from nltk.tokenize import sent_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#stopwordsasatrawi
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()


file_input = open("testing", "r",encoding="utf8")
kalimat = file_input.read()
lower_case = kalimat.lower()
#url
r_url = re.sub(r'\\u\w\w\w\w', '', lower_case)
r_url1=re.sub(r'http\S+','',r_url)
#username
r_username = re.sub('@[^\s]+','',r_url1)
#tagger
r_tagger = re.sub(r'#([^\s]+)', r'\1', r_username)
#tandabaca
def hapus_tanda(instagram): 
			tanda_baca = set(string.punctuation)
			instagram = ''.join(ch for ch in instagram if ch not in tanda_baca)
			return instagram
r_baca=hapus_tanda(r_tagger)
#angka
r_angka = re.sub(r'\w*\d\w*', '',r_baca).strip()

#katadoble
def hapus_katadouble(s):
		    #look for 2 or more repetitions of character and replace with the character itself
		    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
		    return pattern.sub(r"\1\1", s)
r_doble=hapus_katadouble(r_angka)
#pisah
r_doble = r_doble.translate(str.maketrans('','',string.punctuation)).lower()
 
tokens = nltk.tokenize.word_tokenize(r_doble)
#kemunculan
kemunculan = nltk.FreqDist(tokens)
print(kemunculan.most_common())
kemunculan.plot(30,cumulative=False)
plt.show()
#Tokenizing: Sentences Tokenizing Using NLTK Module
tokens1 = nltk.tokenize.sent_tokenize(r_doble)
print(tokens1)
# print(tokens)
#Filtering using Sastrawi: Stopword List
stop = stopword.remove(r_doble)
tokens2 = nltk.tokenize.word_tokenize(stop)

#stemming sastrawi
hasil = stemmer.stem(r_doble)
print(hasil)
# print(tokens2)
#simpan ke file txt, output berupa file txt yang disimpan pada direktori stemming (silahkan buat direktori stemming
#pada laptop/komputer anda
file_output = open("CLEANING DATA/output_KUNTILANAK", "w")

# tulis teks ke file
file_output.write(hasil)

# tutup file
file_output.close()

file_input.close()
