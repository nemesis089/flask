from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

app = Flask(__name__)

@app.route('/api', methods = ['GET'])
def returnascii():
    d = {}
    inputchr = str(request.args['query'])
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="bod_Tibt", tgt_lang='eng_Latn', max_length = 400)
    answer = (translator(inputchr))
    d['output'] = answer
    return d

if __name__ =="__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
