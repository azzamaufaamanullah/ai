import json
import random

data = []

questions = [
    "Siapa kamu?",
    "Apa tujuanmu?",
    "Kamu bisa apa?",
    "Siapa yang membuatmu?",
    "Apa itu AI?",
    "Apa fungsi AI?",
    "Apa pekerjaanmu?",
    "Kamu tinggal dimana?",
    "Apa yang kamu lakukan?",
    "Apakah kamu pintar?"
]

answers = [
    "Saya adalah AI buatan Azzam.",
    "Tujuan saya adalah membantu manusia.",
    "Saya bisa menjawab pertanyaan dan membantu tugas.",
    "Saya dibuat oleh Azzam.",
    "AI adalah kecerdasan buatan.",
    "AI membantu manusia dalam berbagai tugas.",
    "Saya bekerja sebagai asisten virtual.",
    "Saya tidak memiliki lokasi fisik.",
    "Saya membantu menjawab pertanyaan.",
    "Saya terus belajar menjadi lebih baik."
]

# 🔥 generate 150 data
for _ in range(150):
    q = random.choice(questions)
    a = random.choice(answers)
    
    data.append({
        "instruction": q,
        "output": a
    })

# simpan
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ dataset berhasil dibuat:", len(data))