from transformers import pipeline
model = pipeline('text-classification',model='./train_dir/checkpoint-669')
print(model('我真的无语'))