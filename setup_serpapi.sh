#!/bin/bash

echo "==============================================="
echo "SerpAPI Setup for TruthScore"
echo "==============================================="
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

echo "1. Installing SerpAPI Python package..."
pip install google-search-results==2.4.2

echo ""
echo "2. SerpAPI Configuration Instructions:"
echo ""
echo "   a) Go to https://serpapi.com/manage-api-key"
echo "   b) Sign up for a free account (100 searches/month)"
echo "   c) Copy your API key from the dashboard"
echo "   d) Set the environment variable:"
echo ""
echo "      export SERPAPI_KEY=\"your_api_key_here\""
echo ""
echo "   e) Optional: Add to your shell profile (~/.bashrc or ~/.zshrc):"
echo "      echo 'export SERPAPI_KEY=\"your_api_key_here\"' >> ~/.bashrc"
echo ""
echo "3. Optional Configuration (Environment Variables):"
echo ""
echo "   SERPAPI_DAILY_LIMIT=10        # Daily search limit (default: 10)"
echo "   SERPAPI_MONTHLY_LIMIT=90      # Monthly search limit (default: 90)"
echo "   SERPAPI_DELAY_SECONDS=1.5     # Delay between searches (default: 1.5)"
echo ""
echo "4. Test your setup:"
echo ""
echo "   python -c \"import os; from serpapi import GoogleSearch; print('SerpAPI Key:', 'SET' if os.getenv('SERPAPI_KEY') else 'NOT SET')\""
echo ""
echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo ""
echo "Note: The free tier provides 100 searches/month."
echo "TruthScore is configured to use 2 searches per claim (max 2 claims)"
echo "so you can analyze ~25 videos per month with web research."
echo "" 