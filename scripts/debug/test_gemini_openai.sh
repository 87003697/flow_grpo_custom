PYTHONPATH=. python scripts/debug/test_gemini_openai.py \
    --encoder-type group \
    --api-source 3 \
    --prompt-version v2 \
    --max-concurrent 50 \
    --max-tokens 10000 \
    --thinking \
    --debug-response \
    --batch 5