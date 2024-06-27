from flask import Flask, request, render_template
import spacy
from gensim import corpora, models
from googletrans import Translator

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
translator = Translator()

def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def topic_modeling(text):
    texts = [[word.lemma_ for word in nlp(text) if not word.is_stop and not word.is_punct]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)
    return topics

def translate_text(text, dest_language='fr'):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_text = request.form['input_text']
    pos_tags = pos_tagging(input_text)
    entities = ner(input_text)
    topics = topic_modeling(input_text)
    translated_text = translate_text(input_text)
    
    return render_template('result.html', input_text=input_text, pos_tags=pos_tags, entities=entities, topics=topics, translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
