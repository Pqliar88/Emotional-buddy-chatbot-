# Emotion-First Chatbot Backend

A privacy-first, emotion-aware chatbot backend combining:
- Local ML emotion models (text + voice)
- Cloud LLMs (provider-agnostic via env config)
- Safety-first emotional arbitration logic

## Architecture

User → Frontend → Flask API
├─ Text Emotion Model (local)
├─ Voice Emotion Model (local)
└─ LLM chatbot


## Setup

### 1. Clone
```bash
git clone https://github.com/Pqliar88/Emotional-buddy-chatbot-.git
cd Emotional-buddy-chatbot-
```
### 2. Create venv
```
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Configure env
```
cp .env.example .env
```
Edit .env and set your LLM provider credentials.

### 5. Run the server
```
python app.py
```
Server will start at:
```
http://localhost:5000
```

## API Endpoints
```
POST /api/register
POST /api/chat
POST /api/analyze/voice
GET /api/health
```
## Notes
```
Audio uploads require ffmpeg
Models are loaded once at startup
Session data is in-memory (stateless by design)
```
## License

MIT
