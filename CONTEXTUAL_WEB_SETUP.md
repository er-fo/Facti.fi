# ContextualWeb Search API Setup Guide

## Overview
This application now uses the **ContextualWeb Search API** via RapidAPI to replace the broken DuckDuckGo web search functionality. This provides reliable, high-quality web search results for fact-checking claims.

## Why ContextualWeb Search API?
- **Reliable**: No rate limiting issues like DuckDuckGo
- **High Quality**: AI-powered search results optimized for accuracy
- **Cost Effective**: 100 free requests per day, then $5 per 1000 requests
- **Professional**: Proper API with authentication and rate limiting
- **Stable**: Hosted on RapidAPI platform with SLA guarantees

## Setup Instructions

### Step 1: Get Your API Key

1. **Sign up for RapidAPI**:
   - Go to [https://rapidapi.com/](https://rapidapi.com/)
   - Click "Sign Up" and create a free account
   - You can use Google, GitHub, or Facebook to sign up

2. **Subscribe to ContextualWeb Search API**:
   - Navigate to: [ContextualWeb Search API on RapidAPI](https://rapidapi.com/contextualwebsearch/api/web-search/)
   - Click "Subscribe to Test"
   - Select the **Basic Plan** (Free - 100 requests/day)
   - Click "Select Plan"

3. **Get Your API Key**:
   - After subscribing, you'll see your API key in the "Header Parameters" section
   - It will look like: `x-rapidapi-key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - Copy this key value (not including the "x-rapidapi-key:" part)

### Step 2: Configure Environment Variables

Add the following environment variable to your system:

```bash
export CONTEXTUAL_WEB_API_KEY="your_api_key_here"
```

#### For macOS/Linux:
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export CONTEXTUAL_WEB_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

#### For Windows:
Set environment variable through System Properties or use PowerShell:
```powershell
[Environment]::SetEnvironmentVariable("CONTEXTUAL_WEB_API_KEY", "your_api_key_here", "User")
```

#### Alternative: Create a `.env` file
Create a `.env` file in your project root:
```
CONTEXTUAL_WEB_API_KEY=your_api_key_here
```

### Step 3: Optional Configuration

You can also set these optional environment variables:

```bash
# Daily quota limit (default: 90, leaving buffer from 100 free limit)
export CONTEXTUAL_WEB_QUOTA_LIMIT=90

# Request timeout in seconds (default: 15)
export CONTEXTUAL_WEB_TIMEOUT=15
```

### Step 4: Test the Integration

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Check health endpoint**:
   Visit `http://localhost:8000/health` to see web search status:
   ```json
   {
     "web_search": {
       "status": "available",
       "message": "Available (90 requests remaining)",
       "api_configured": true,
       "quota_used": 0,
       "quota_remaining": 90,
       "quota_limit": 90,
       "quota_date": "2024-01-15"
     }
   }
   ```

3. **Test with a transcript analysis**:
   - Use the web interface to analyze any video
   - Check the logs for messages like:
     ```
     ContextualWeb API returned 5 results for query: ...
     Web research completed for claim 1: VERIFIED (Score: 85/100, Sources: 15, Quota used: 3)
     ```

## Pricing Plans

### Free Tier (Basic Plan)
- **Cost**: Free
- **Requests**: 100 per day
- **Features**: Full web search access
- **Perfect for**: Testing and small applications

### Paid Tiers (if you need more)
- **Pro Plan**: $5 per 1000 requests
- **Unlimited**: Contact RapidAPI for enterprise pricing

## Quota Management

The application automatically:
- **Tracks daily quota usage** in `web_search_quota.json`
- **Resets quota** automatically each day
- **Prevents overages** by checking quota before each request
- **Falls back to AI-only analysis** when quota is exhausted
- **Adds delays** between requests to be respectful to the API

## Monitoring

### Check Quota Status
Visit `/health` endpoint to see current quota usage.

### Log Messages
Monitor logs for these messages:
- `✓ ContextualWeb Search API available`
- `✓ ContextualWeb API returned X results for query`
- `⚠ Quota exhausted, skipping remaining queries`
- `❌ ContextualWeb API authentication failed - check API key`

### Common Issues

1. **"API key not configured"**
   - Set the `CONTEXTUAL_WEB_API_KEY` environment variable
   - Restart the application

2. **"Authentication failed - check API key"**
   - Verify your API key is correct
   - Ensure you're subscribed to the API on RapidAPI

3. **"Daily quota exhausted"**
   - Wait until tomorrow for quota reset
   - Consider upgrading to a paid plan
   - The app will continue working with AI-only analysis

4. **"Rate limit exceeded"**
   - The app includes built-in delays, but if this occurs:
   - Reduce the number of search queries per claim
   - Increase delays between requests

## Advantages Over DuckDuckGo

| Feature | DuckDuckGo | ContextualWeb API |
|---------|------------|-------------------|
| Reliability | ❌ Frequent failures | ✅ 99.9% uptime |
| Rate Limiting | ❌ Aggressive blocking | ✅ Predictable limits |
| API Quality | ❌ Unofficial/unstable | ✅ Professional API |
| Error Handling | ❌ Cryptic errors | ✅ Clear error codes |
| Quota Management | ❌ No visibility | ✅ Full tracking |
| Cost | Free but broken | ✅ Free tier + paid options |
| Support | ❌ None | ✅ RapidAPI support |

## Troubleshooting

If web search isn't working:

1. **Check environment variable**:
   ```bash
   echo $CONTEXTUAL_WEB_API_KEY
   ```

2. **Test API key manually**:
   ```bash
   curl -X GET \
     "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/WebSearchAPI?q=test&pageNumber=1&pageSize=5&autoCorrect=true" \
     -H "x-rapidapi-key: YOUR_API_KEY" \
     -H "x-rapidapi-host: contextualwebsearch-websearch-v1.p.rapidapi.com"
   ```

3. **Check quota file**:
   - Look for `web_search_quota.json` in project root
   - Delete it to reset quota tracking if needed

4. **Review logs**:
   - Check `logs/` directory for detailed error messages
   - Look for ContextualWeb API related errors

## Migration Notes

This replaces the old DuckDuckGo implementation with:
- ✅ Reliable web search functionality
- ✅ Professional API with SLA
- ✅ Built-in quota management
- ✅ Better error handling
- ✅ Enhanced monitoring
- ✅ Long-term sustainability

The application will automatically fall back to AI-only analysis if web search is unavailable, ensuring continuous operation. 