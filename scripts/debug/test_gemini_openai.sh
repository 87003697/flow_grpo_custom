PYTHONPATH=. python scripts/debug/test_gemini_openai.py \
    --api-source 3 \
    --prompt-version v2 \
    --max-concurrent 50 \
    --max-tokens 1000 \
    --thinking \
    --debug-response \
    --batch 5