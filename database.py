import sqlite3
import json
import os
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import hashlib

class AnalysisDatabase:
    def __init__(self, db_path='analyses.db'):
        """Initialize the analysis database"""
        self.db_path = db_path
        self.logger = logging.getLogger('app')
        self._init_database()
    
    def _init_database(self):
        """Initialize the database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        normalized_url TEXT NOT NULL,
                        url_hash TEXT NOT NULL UNIQUE,
                        title TEXT,
                        analysis_data TEXT NOT NULL,
                        transcript_data TEXT,
                        credibility_score INTEGER,
                        analysis_type TEXT DEFAULT 'comprehensive',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create index for faster lookups
                conn.execute('CREATE INDEX IF NOT EXISTS idx_url_hash ON analyses(url_hash)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_normalized_url ON analyses(normalized_url)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at)')
                
                conn.commit()
                self.logger.info("Analysis database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def _normalize_url(self, url):
        """Normalize URL to ensure consistent comparison"""
        try:
            # Parse the URL
            parsed = urlparse(url.lower().strip())
            
            # Remove common tracking parameters
            tracking_params = {
                't', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'ref', 'source', 'si', 'feature', 'app', 'via'
            }
            
            # Parse query parameters
            query_params = parse_qs(parsed.query)
            
            # Filter out tracking parameters
            clean_params = {k: v for k, v in query_params.items() 
                          if k.lower() not in tracking_params}
            
            # Rebuild query string
            clean_query = urlencode(clean_params, doseq=True)
            
            # Remove common URL variations
            netloc = parsed.netloc.replace('www.', '').replace('m.', '')
            
            # Handle YouTube URL variations
            if 'youtube.com' in netloc or 'youtu.be' in netloc:
                if 'youtu.be' in netloc:
                    # Convert youtu.be/VIDEO_ID to youtube.com/watch?v=VIDEO_ID
                    video_id = parsed.path.lstrip('/')
                    netloc = 'youtube.com'
                    path = '/watch'
                    clean_query = f'v={video_id}'
                else:
                    path = parsed.path
                    # Ensure we have the video ID for YouTube
                    if 'v' in clean_params:
                        clean_query = f"v={clean_params['v'][0]}"
            else:
                path = parsed.path.rstrip('/')
            
            # Reconstruct the normalized URL
            normalized = urlunparse((
                parsed.scheme or 'https',
                netloc,
                path,
                '',  # params
                clean_query,
                ''   # fragment
            ))
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Error normalizing URL {url}: {e}")
            return url.lower().strip()
    
    def _generate_url_hash(self, normalized_url):
        """Generate a hash for the normalized URL"""
        return hashlib.sha256(normalized_url.encode()).hexdigest()[:32]
    
    def store_analysis(self, url, title, analysis_data, transcript_data=None, 
                      credibility_score=None, analysis_type='comprehensive'):
        """Store an analysis result in the database"""
        try:
            normalized_url = self._normalize_url(url)
            url_hash = self._generate_url_hash(normalized_url)
            
            with sqlite3.connect(self.db_path) as conn:
                # Check if analysis already exists
                existing = conn.execute(
                    'SELECT id FROM analyses WHERE url_hash = ?', 
                    (url_hash,)
                ).fetchone()
                
                if existing:
                    # Update existing analysis
                    conn.execute('''
                        UPDATE analyses SET 
                            url = ?, title = ?, analysis_data = ?, transcript_data = ?,
                            credibility_score = ?, analysis_type = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE url_hash = ?
                    ''', (url, title, json.dumps(analysis_data), 
                         json.dumps(transcript_data) if transcript_data else None,
                         credibility_score, analysis_type, url_hash))
                    
                    analysis_id = existing[0]
                    self.logger.info(f"Updated existing analysis {analysis_id} for URL: {url}")
                else:
                    # Insert new analysis
                    cursor = conn.execute('''
                        INSERT INTO analyses 
                        (url, normalized_url, url_hash, title, analysis_data, transcript_data, 
                         credibility_score, analysis_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (url, normalized_url, url_hash, title, json.dumps(analysis_data),
                         json.dumps(transcript_data) if transcript_data else None,
                         credibility_score, analysis_type))
                    
                    analysis_id = cursor.lastrowid
                    self.logger.info(f"Stored new analysis {analysis_id} for URL: {url}")
                
                conn.commit()
                return analysis_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error storing analysis for URL {url}: {e}")
            raise
    
    def find_existing_analysis(self, url):
        """Find existing analysis for a URL"""
        try:
            normalized_url = self._normalize_url(url)
            url_hash = self._generate_url_hash(normalized_url)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                result = conn.execute('''
                    SELECT id, url, title, analysis_data, transcript_data, 
                           credibility_score, analysis_type, created_at, updated_at
                    FROM analyses 
                    WHERE url_hash = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                ''', (url_hash,)).fetchone()
                
                if result:
                    return {
                        'id': result['id'],
                        'url': result['url'],
                        'title': result['title'],
                        'analysis_data': json.loads(result['analysis_data']),
                        'transcript_data': json.loads(result['transcript_data']) if result['transcript_data'] else None,
                        'credibility_score': result['credibility_score'],
                        'analysis_type': result['analysis_type'],
                        'created_at': result['created_at'],
                        'updated_at': result['updated_at']
                    }
                
                return None
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.error(f"Error finding analysis for URL {url}: {e}")
            return None
    
    def get_all_analyses(self, limit=50, offset=0, order_by='updated_at DESC'):
        """Get all analyses with pagination"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get total count
                total_count = conn.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
                
                # Get analyses with pagination
                results = conn.execute(f'''
                    SELECT id, url, title, credibility_score, analysis_type, 
                           created_at, updated_at
                    FROM analyses 
                    ORDER BY {order_by}
                    LIMIT ? OFFSET ?
                ''', (limit, offset)).fetchall()
                
                analyses = []
                for row in results:
                    analyses.append({
                        'id': row['id'],
                        'url': row['url'],
                        'title': row['title'],
                        'credibility_score': row['credibility_score'],
                        'analysis_type': row['analysis_type'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    })
                
                return {
                    'analyses': analyses,
                    'total_count': total_count,
                    'has_more': (offset + limit) < total_count
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting all analyses: {e}")
            return {'analyses': [], 'total_count': 0, 'has_more': False}
    
    def get_analysis_by_id(self, analysis_id):
        """Get a specific analysis by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                result = conn.execute('''
                    SELECT id, url, title, analysis_data, transcript_data, 
                           credibility_score, analysis_type, created_at, updated_at
                    FROM analyses 
                    WHERE id = ?
                ''', (analysis_id,)).fetchone()
                
                if result:
                    return {
                        'id': result['id'],
                        'url': result['url'],
                        'title': result['title'],
                        'analysis_data': json.loads(result['analysis_data']),
                        'transcript_data': json.loads(result['transcript_data']) if result['transcript_data'] else None,
                        'credibility_score': result['credibility_score'],
                        'analysis_type': result['analysis_type'],
                        'created_at': result['created_at'],
                        'updated_at': result['updated_at']
                    }
                
                return None
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.error(f"Error getting analysis {analysis_id}: {e}")
            return None
    
    def delete_analysis(self, analysis_id):
        """Delete an analysis by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Deleted analysis {analysis_id}")
                    return True
                else:
                    self.logger.warning(f"Analysis {analysis_id} not found for deletion")
                    return False
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting analysis {analysis_id}: {e}")
            return False
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                total_analyses = conn.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
                
                recent_analyses = conn.execute('''
                    SELECT COUNT(*) FROM analyses 
                    WHERE created_at >= datetime('now', '-7 days')
                ''').fetchone()[0]
                
                avg_score = conn.execute('''
                    SELECT AVG(credibility_score) FROM analyses 
                    WHERE credibility_score IS NOT NULL
                ''').fetchone()[0]
                
                analysis_types = conn.execute('''
                    SELECT analysis_type, COUNT(*) as count 
                    FROM analyses 
                    GROUP BY analysis_type
                ''').fetchall()
                
                return {
                    'total_analyses': total_analyses,
                    'recent_analyses': recent_analyses,
                    'average_credibility_score': round(avg_score, 1) if avg_score else 0,
                    'analysis_types': dict(analysis_types)
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {
                'total_analyses': 0,
                'recent_analyses': 0,
                'average_credibility_score': 0,
                'analysis_types': {}
            } 