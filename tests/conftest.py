import os

# Provide dummy env so `settings` import doesn't fail in tests without a real .env
os.environ.setdefault("GROQ_API_KEY", "test_key_1234567890")
os.environ.setdefault("PINECONE_API_KEY", "test_key_1234567890")
