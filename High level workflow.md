1. Input of link - and extra parameters from users (if existant)
2. Download of (only audio) speech/video via simply python script
3. Process audio file with local OpenAI Whisper model
4. Whisper returns transcript with timestamps for each segment
(5). Eventual extra parameters get put in prompt together with transcript
6. Transcript gets analysed by LLM 
    5.2 Eventual web research gets performed
7. LLM returns score with considerations. 