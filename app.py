from flask import Flask, request, jsonify
import re
import string
import torch
from torch import clamp
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import os
import logging

# Inisialisasi logging
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Ambil token API dari variabel lingkungan
API_TOKEN = "GNyft8OsvZAlizusJeSG1I8RxyErxygHBTzKGW8dllZIADvacj"

# Kelas Token Similarity
class TokenSimilarity:

    def __init__(self, model_path="indobenchmark/indobert-base-p1"):
        # Inisialisasi tokenizer dan model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __cleaning(self, text):
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __process(self, first_token, second_token, max_length, truncation, padding):
        inputs = self.tokenizer([first_token, second_token],
                                max_length=max_length,
                                truncation=truncation,
                                padding=padding,
                                return_tensors='pt')

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        attention = inputs["attention_mask"]
        outputs = self.model(**inputs)

        # Dapatkan embeddings dengan mean pooling
        embeddings = outputs.last_hidden_state
        attention = attention.unsqueeze(-1).to(torch.float32)
        masked_embeddings = embeddings * attention
        summed = masked_embeddings.sum(1)
        counts = clamp(attention.sum(1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.detach().cpu().numpy()

    def predict(self, first_token, second_token, max_length=128, truncation=True, padding="max_length"):
        first_token = self.__cleaning(first_token)
        second_token = self.__cleaning(second_token)

        embeddings = self.__process(first_token, second_token, max_length, truncation, padding)

        # Hitung cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

# Inisialisasi Flask
app = Flask(__name__)

# Aktifkan CORS hanya untuk domain tertentu
CORS(app, resources={r"/*": {"origins": "https://yourdomain.com"}})

# Inisialisasi TokenSimilarity
ts = TokenSimilarity()

@app.route('/check_similarity', methods=['POST'])
def predict_similarity():
    try:
        # Validasi API token
        auth_token = request.headers.get("Authorization")
        if auth_token != f"Bearer {API_TOKEN}":
            logging.warning("Akses tidak sah.")
            return jsonify({"error": "Unauthorized"}), 401

        # Ambil data JSON
        data = request.json
        student_answer = data.get("studentAnswer", "")
        key_answer = data.get("keyAnswer", "")

        if not student_answer or not key_answer:
            logging.error("Jawaban siswa atau kunci jawaban kosong.")
            return jsonify({"error": "Jawaban dan kunci jawaban harus dimasukkan."}), 400

        # Hitung skor kemiripan
        similarity_score = ts.predict(student_answer, key_answer)
        logging.info("Request berhasil diproses.")
        return jsonify({"similarityScore": round(float(similarity_score), 4)})

    except Exception as e:
        logging.error(f"Terjadi error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan aplikasi
    app.run(host="0.0.0.0", port=5000, debug=False)
