# SerpAPI Integration for TruthScore

## Overview

TruthScore now uses SerpAPI (Google Search API) for reliable web research instead of the previous DuckDuckGo scraping approach. This provides:

- ✅ **Reliable searches** - Official API instead of scraping
- ✅ **Rate limiting protection** - Smart quota management
- ✅ **Better results quality** - Google's search algorithms
- ✅ **Transparent reporting** - Clear indication when web research actually worked

## Key Improvements

### Before (DuckDuckGo Scraping)
- ❌ Frequent rate limiting errors
- ❌ Cascading search failures
- ❌ Unreliable scraping library
- ❌ No usage tracking

### After (SerpAPI)
- ✅ Controlled API usage with quota tracking
- ✅ Intelligent rate limiting (1.5s delays)
- ✅ Graceful degradation when quota exceeded
- ✅ Clear distinction between web-researched vs AI-only claims

## Technical Changes

### Search Volume Optimization
- **Claims per analysis**: Reduced from 3 to 2 claims
- **Queries per claim**: Reduced from 3 to 2 queries
- **Results per query**: Limited to 3 results
- **Total searches per analysis**: Maximum 4 searches (vs 9 previously)

### Quota Management
- **Daily limit**: 10 searches (configurable)
- **Monthly limit**: 90 searches (leaves 10 buffer from 100/month free tier)
- **Usage tracking**: Persistent tracking across restarts
- **Automatic cleanup**: Old usage data cleaned automatically

### Rate Limiting
- **Delay between searches**: 1.5 seconds (configurable)
- **Quota checks**: Before each search operation
- **Graceful fallback**: AI-only analysis when quota exceeded

## Configuration

### Required Environment Variables
```bash
export SERPAPI_KEY="your_api_key_here"
```

### Optional Environment Variables
```bash
export SERPAPI_DAILY_LIMIT=10        # Daily search limit
export SERPAPI_MONTHLY_LIMIT=90      # Monthly search limit  
export SERPAPI_DELAY_SECONDS=1.5     # Delay between searches
```

## Usage Estimates

With the free tier (100 searches/month):
- **Maximum analyses with web research**: ~25 per month
- **Searches per analysis**: 4 (2 claims × 2 queries each)
- **Conservative daily limit**: 10 searches = ~2-3 analyses per day

## API Response Format

SerpAPI returns structured data:
```json
{
  "organic_results": [
    {
      "title": "Result title",
      "snippet": "Result description", 
      "link": "https://example.com",
      "position": 1
    }
  ]
}
```

## Error Handling

### Quota Exceeded
- Logs clear warning message
- Falls back to AI-only analysis
- Updates research method in results

### API Errors
- Individual search failures don't break entire analysis
- Retry logic for temporary failures
- Clear error reporting

### No Results Found
- Graceful handling of empty search results
- Falls back to AI-only analysis for affected claims
- Maintains analysis quality

## Monitoring

### Health Endpoint
The `/health` endpoint now reports:
- SerpAPI availability status
- Current quota status
- Usage statistics

### Logging
Enhanced logging includes:
- Quota status before/after searches
- Research method used for each claim
- Clear distinction between web vs AI-only results

## Migration Notes

### Removed Dependencies
- `duckduckgo-search==6.3.5` → Removed

### Added Dependencies  
- `google-search-results==2.4.2` → Added

### Data Format Changes
- Search results now include `position` field
- Research results include `search_result_positions`
- Enhanced metadata for research method tracking 