# sentence = """Thomas Jefferson began building Monticello at the age of 26."""
# sentence += """Construction was done mostly by local masons and carpenters.\n"""
# sentence += """He moved into the South Pavilion in 1770.\n"""
# sentence += """Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."""

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# a = lemmatizer.lemmatize('better')
b = lemmatizer.lemmatize('happiest', pos='a')
print(b)

