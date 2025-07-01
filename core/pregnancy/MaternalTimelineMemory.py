# =====================================
# FILE: core/pregnancy/MaternalTimelineMemory.py
# Phase 1.2: Specialized Memory Systems for Maternal Care
# =====================================

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from ..MemoryCore import MemoryCore

@dataclass
class MaternalMilestone:
    """Represents a significant maternal journey milestone"""
    milestone_id: str
    milestone_type: str  # 'medical', 'emotional', 'physical', 'planning'
    week_number: int
    title: str
    description: str
    significance_score: float  # 0.0 to 1.0
    emotional_impact: str
    date_occurred: str
    user_notes: str = ""
    medical_relevance: bool = False
    shareable: bool = True
    tags: List[str] = None

@dataclass
class PregnancyMemoryContext:
    """Rich context for pregnancy-related memories"""
    conception_date: str
    due_date: str
    current_week: int
    trimester: int
    recent_milestones: List[MaternalMilestone]
    health_patterns: Dict[str, Any]
    emotional_trends: Dict[str, Any]
    upcoming_appointments: List[Dict]

class MaternalTimelineMemory(MemoryCore):
    """
    Enhanced memory system specifically designed for maternal journey
    Builds on your existing MemoryCore with pregnancy-specific intelligence
    """
    
    def __init__(self, memory_file: str = "maternal_memory.enc", key_file: str = "maternal.key"):
        super().__init__(memory_file, key_file)
        self.milestone_patterns = self._load_milestone_patterns()
        self._init_maternal_tables()
        
    def _load_milestone_patterns(self) -> Dict[str, Any]:
        """Load patterns for automatic milestone detection"""
        return {
            "medical_milestones": {
                "first_heartbeat": {
                    "keywords": ["heartbeat", "heart beat", "first time hearing"],
                    "typical_weeks": [6, 7, 8, 9, 10],
                    "significance": 0.9,
                    "emotional_impact": "overwhelming_joy"
                },
                "first_movement": {
                    "keywords": ["movement", "kick", "flutter", "baby moving"],
                    "typical_weeks": [16, 17, 18, 19, 20, 21, 22],
                    "significance": 0.9,
                    "emotional_impact": "connection"
                },
                "anatomy_scan": {
                    "keywords": ["anatomy scan", "20 week scan", "ultrasound"],
                    "typical_weeks": [18, 19, 20, 21, 22],
                    "significance": 0.8,
                    "emotional_impact": "anticipation"
                },
                "glucose_test": {
                    "keywords": ["glucose test", "gestational diabetes"],
                    "typical_weeks": [24, 25, 26, 27, 28],
                    "significance": 0.6,
                    "emotional_impact": "anxiety"
                },
                "third_trimester": {
                    "keywords": ["third trimester", "28 weeks"],
                    "typical_weeks": [28],
                    "significance": 0.7,
                    "emotional_impact": "anticipation"
                }
            },
            "emotional_milestones": {
                "telling_family": {
                    "keywords": ["told family", "announced", "shared news"],
                    "typical_weeks": [8, 9, 10, 11, 12],
                    "significance": 0.8,
                    "emotional_impact": "joy"
                },
                "first_purchase": {
                    "keywords": ["first baby", "bought", "purchased", "shopping"],
                    "typical_weeks": [12, 13, 14, 15, 16],
                    "significance": 0.6,
                    "emotional_impact": "excitement"
                },
                "nursery_setup": {
                    "keywords": ["nursery", "baby room", "decorating"],
                    "typical_weeks": [25, 26, 27, 28, 29, 30],
                    "significance": 0.7,
                    "emotional_impact": "nesting"
                },
                "hospital_bag": {
                    "keywords": ["hospital bag", "packed", "ready"],
                    "typical_weeks": [35, 36, 37, 38],
                    "significance": 0.8,
                    "emotional_impact": "preparation"
                }
            },
            "physical_milestones": {
                "morning_sickness_start": {
                    "keywords": ["nausea", "morning sickness", "throwing up"],
                    "typical_weeks": [4, 5, 6, 7, 8],
                    "significance": 0.5,
                    "emotional_impact": "discomfort"
                },
                "energy_return": {
                    "keywords": ["energy back", "feeling better", "second trimester"],
                    "typical_weeks": [13, 14, 15, 16],
                    "significance": 0.6,
                    "emotional_impact": "relief"
                },
                "showing": {
                    "keywords": ["showing", "belly", "bump", "people notice"],
                    "typical_weeks": [16, 17, 18, 19, 20],
                    "significance": 0.7,
                    "emotional_impact": "pride"
                }
            }
        }
    
    def _init_maternal_tables(self):
        """Initialize maternal-specific memory tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Milestones table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maternal_milestones (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    milestone_type TEXT,
                    week_number INTEGER,
                    title TEXT,
                    description TEXT,
                    significance_score REAL,
                    emotional_impact TEXT,
                    date_occurred TEXT,
                    user_notes TEXT,
                    medical_relevance BOOLEAN,
                    shareable BOOLEAN,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pregnancy context table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pregnancy_context (
                    user_id TEXT PRIMARY KEY,
                    conception_date TEXT,
                    due_date TEXT,
                    current_week INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Memory associations table (link memories to pregnancy context)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maternal_memory_associations (
                    memory_id TEXT,
                    milestone_id TEXT,
                    association_type TEXT,
                    strength REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def store_maternal_memory(self, content: str, memory_type: str, user_id: str, 
                            pregnancy_week: int = 0, emotional_context: Dict = None) -> str:
        """
        Store memory with maternal context and automatic milestone detection
        """
        
        # Store base memory using parent class
        memory_id = self.save_fact(content, [], source="maternal_care")
        
        # Detect potential milestones
        detected_milestones = self._detect_milestones(content, pregnancy_week)
        
        # Store milestones
        for milestone in detected_milestones:
            milestone_id = self._store_milestone(milestone, user_id)
            self._associate_memory_milestone(memory_id, milestone_id, "detected")
        
        # Update pregnancy context
        self._update_pregnancy_context(user_id, pregnancy_week)
        
        # Store emotional context if provided
        if emotional_context:
            self._store_emotional_context(memory_id, emotional_context)
        
        print(f"[🤱 MATERNAL MEMORY] Stored with {len(detected_milestones)} milestones detected")
        return memory_id
    
    def _detect_milestones(self, content: str, pregnancy_week: int) -> List[MaternalMilestone]:
        """Automatically detect milestones from content"""
        content_lower = content.lower()
        detected_milestones = []
        
        for category, milestones in self.milestone_patterns.items():
            for milestone_name, patterns in milestones.items():
                
                # Check if keywords match
                keyword_matches = sum(1 for keyword in patterns["keywords"] 
                                    if keyword in content_lower)
                
                if keyword_matches > 0:
                    # Check if week is appropriate
                    week_appropriate = (pregnancy_week == 0 or 
                                      pregnancy_week in patterns["typical_weeks"])
                    
                    if week_appropriate or keyword_matches >= 2:
                        milestone = MaternalMilestone(
                            milestone_id=f"{milestone_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            milestone_type=category.replace('_milestones', ''),
                            week_number=pregnancy_week,
                            title=milestone_name.replace('_', ' ').title(),
                            description=f"Detected from user input: {content[:100]}...",
                            significance_score=patterns["significance"],
                            emotional_impact=patterns["emotional_impact"],
                            date_occurred=datetime.now().isoformat(),
                            tags=[milestone_name, category]
                        )
                        detected_milestones.append(milestone)
        
        return detected_milestones
    
    def _store_milestone(self, milestone: MaternalMilestone, user_id: str) -> str:
        """Store a milestone in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO maternal_milestones 
                (id, user_id, milestone_type, week_number, title, description, 
                 significance_score, emotional_impact, date_occurred, user_notes, 
                 medical_relevance, shareable, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                milestone.milestone_id, user_id, milestone.milestone_type,
                milestone.week_number, milestone.title, milestone.description,
                milestone.significance_score, milestone.emotional_impact,
                milestone.date_occurred, milestone.user_notes,
                milestone.medical_relevance, milestone.shareable,
                json.dumps(milestone.tags or [])
            ))
            conn.commit()
        
        return milestone.milestone_id
    
    def _associate_memory_milestone(self, memory_id: str, milestone_id: str, 
                                  association_type: str, strength: float = 1.0):
        """Associate a memory with a milestone"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO maternal_memory_associations 
                (memory_id, milestone_id, association_type, strength)
                VALUES (?, ?, ?, ?)
            ''', (memory_id, milestone_id, association_type, strength))
            conn.commit()
    
    def get_maternal_context(self, user_id: str) -> PregnancyMemoryContext:
        """Get comprehensive maternal context for memory retrieval"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get pregnancy context
            cursor.execute("SELECT * FROM pregnancy_context WHERE user_id = ?", (user_id,))
            pregnancy_data = cursor.fetchone()
            
            if not pregnancy_data:
                return None
            
            conception_date, due_date, current_week = pregnancy_data[1:4]
            trimester = self._calculate_trimester(current_week)
            
            # Get recent milestones
            cursor.execute('''
                SELECT * FROM maternal_milestones 
                WHERE user_id = ? 
                ORDER BY date_occurred DESC 
                LIMIT 10
            ''', (user_id,))
            
            milestone_rows = cursor.fetchall()
            recent_milestones = []
            
            for row in milestone_rows:
                milestone = MaternalMilestone(
                    milestone_id=row[0],
                    milestone_type=row[2],
                    week_number=row[3],
                    title=row[4],
                    description=row[5],
                    significance_score=row[6],
                    emotional_impact=row[7],
                    date_occurred=row[8],
                    user_notes=row[9] or "",
                    medical_relevance=bool(row[10]),
                    shareable=bool(row[11]),
                    tags=json.loads(row[12] or "[]")
                )
                recent_milestones.append(milestone)
        
        return PregnancyMemoryContext(
            conception_date=conception_date,
            due_date=due_date,
            current_week=current_week,
            trimester=trimester,
            recent_milestones=recent_milestones,
            health_patterns=self._analyze_health_patterns(user_id),
            emotional_trends=self._analyze_emotional_trends(user_id),
            upcoming_appointments=self._get_upcoming_appointments(user_id)
        )
    
    def search_maternal_memories(self, query: str, user_id: str, 
                                context_filter: Dict = None) -> List[Dict]:
        """Search memories with maternal context awareness"""
        
        # Get maternal context
        maternal_context = self.get_maternal_context(user_id)
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(query, maternal_context, context_filter)
        
        # Search base memories
        base_results = self.search_memory(enhanced_query, limit=50)
        
        # Enhance results with maternal context
        enhanced_results = []
        for result in base_results:
            enhanced_result = self._enhance_result_with_context(result, maternal_context)
            enhanced_results.append(enhanced_result)
        
        # Sort by relevance and maternal significance
        enhanced_results.sort(key=lambda x: x.get('maternal_relevance_score', 0), reverse=True)
        
        return enhanced_results[:20]  # Return top 20 results
    
    def _enhance_query_with_context(self, query: str, context: PregnancyMemoryContext, 
                                   context_filter: Dict) -> str:
        """Enhance search query with maternal context"""
        enhanced_terms = [query]
        
        if context:
            # Add week-specific context
            enhanced_terms.append(f"week {context.current_week}")
            enhanced_terms.append(f"trimester {context.trimester}")
            
            # Add recent milestone context
            for milestone in context.recent_milestones[:3]:
                enhanced_terms.extend(milestone.tags or [])
        
        # Apply context filters
        if context_filter:
            if context_filter.get('milestone_type'):
                enhanced_terms.append(context_filter['milestone_type'])
            if context_filter.get('emotional_state'):
                enhanced_terms.append(context_filter['emotional_state'])
            if context_filter.get('trimester'):
                enhanced_terms.append(f"trimester {context_filter['trimester']}")
        
        return " ".join(enhanced_terms)
    
    def _enhance_result_with_context(self, result: Dict, context: PregnancyMemoryContext) -> Dict:
        """Enhance search result with maternal context"""
        enhanced_result = result.copy()
        
        # Calculate maternal relevance score
        maternal_score = 0.0
        
        # Check if memory is associated with milestones
        memory_id = result.get('id', '')
        associated_milestones = self._get_associated_milestones(memory_id)
        
        if associated_milestones:
            maternal_score += 0.4
            enhanced_result['associated_milestones'] = associated_milestones
        
        # Check temporal relevance (memories from similar pregnancy weeks)
        if context and 'week' in result.get('content', '').lower():
            maternal_score += 0.3
        
        # Check emotional relevance
        if context and context.emotional_trends:
            current_emotion = context.emotional_trends.get('current_primary_emotion', '')
            if current_emotion and current_emotion in result.get('content', '').lower():
                maternal_score += 0.3
        
        enhanced_result['maternal_relevance_score'] = maternal_score
        enhanced_result['pregnancy_context'] = context.current_week if context else 0
        
        return enhanced_result
    
    def _get_associated_milestones(self, memory_id: str) -> List[Dict]:
        """Get milestones associated with a memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT m.title, m.emotional_impact, m.significance_score, ma.association_type
                FROM maternal_milestones m
                JOIN maternal_memory_associations ma ON m.id = ma.milestone_id
                WHERE ma.memory_id = ?
            ''', (memory_id,))
            
            results = cursor.fetchall()
            return [{'title': r[0], 'emotional_impact': r[1], 
                    'significance': r[2], 'type': r[3]} for r in results]
    
    def _update_pregnancy_context(self, user_id: str, pregnancy_week: int):
        """Update pregnancy context tracking"""
        if pregnancy_week <= 0:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO pregnancy_context 
                (user_id, current_week, last_updated)
                VALUES (?, ?, ?)
            ''', (user_id, pregnancy_week, datetime.now().isoformat()))
            conn.commit()
    
    def _store_emotional_context(self, memory_id: str, emotional_context: Dict):
        """Store emotional context with memory"""
        # This would integrate with your EmotionCore
        context_data = {
            'memory_id': memory_id,
            'emotional_context': emotional_context,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in your existing emotion tracking system
        # This connects to your EmotionalJournal or similar component
        print(f"[💭 EMOTIONAL CONTEXT] Stored for memory {memory_id}")
    
    def _calculate_trimester(self, pregnancy_week: int) -> int:
        """Calculate trimester from pregnancy week"""
        if pregnancy_week <= 12:
            return 1
        elif pregnancy_week <= 27:
            return 2
        else:
            return 3
    
    def _analyze_health_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze health patterns from stored memories"""
        # This would integrate with your MaternalHealthProfile
        return {
            "energy_trends": "stable",
            "symptom_patterns": ["mild_nausea", "fatigue"],
            "sleep_quality": "improving"
        }
    
    def _analyze_emotional_trends(self, user_id: str) -> Dict[str, Any]:
        """Analyze emotional trends from milestones"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT emotional_impact, week_number, significance_score
                FROM maternal_milestones
                WHERE user_id = ?
                ORDER BY date_occurred DESC
                LIMIT 20
            ''', (user_id,))
            
            results = cursor.fetchall()
            
            if not results:
                return {}
            
            # Analyze emotional patterns
            emotions = [r[0] for r in results]
            recent_emotions = emotions[:5]
            
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            most_common = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
            
            return {
                "current_primary_emotion": recent_emotions[0] if recent_emotions else "neutral",
                "most_common_emotion": most_common,
                "emotional_variety": len(set(emotions)),
                "recent_trend": "positive" if "joy" in recent_emotions[:3] else "mixed"
            }
    
    def _get_upcoming_appointments(self, user_id: str) -> List[Dict]:
        """Get upcoming medical appointments (placeholder for integration)"""
        # This would integrate with calendar or appointment tracking
        return [
            {"type": "prenatal_checkup", "date": "2025-06-15", "week": 32},
            {"type": "ultrasound", "date": "2025-06-22", "week": 33}
        ]
    
    def generate_milestone_summary(self, user_id: str, week_range: Tuple[int, int] = None) -> str:
        """Generate a beautiful summary of maternal milestones"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM maternal_milestones WHERE user_id = ?"
            params = [user_id]
            
            if week_range:
                query += " AND week_number BETWEEN ? AND ?"
                params.extend(week_range)
            
            query += " ORDER BY week_number, date_occurred"
            
            cursor.execute(query, params)
            milestones = cursor.fetchall()
        
        if not milestones:
            return "No milestones recorded yet. Your journey is just beginning! 🌟"
        
        summary = "🌸 Your Maternal Journey Milestones 🌸\n\n"
        
        current_trimester = 0
        for milestone in milestones:
            week = milestone[3]
            trimester = self._calculate_trimester(week)
            
            if trimester != current_trimester:
                current_trimester = trimester
                summary += f"\n✨ TRIMESTER {trimester} ✨\n"
            
            title = milestone[4]
            description = milestone[5]
            emotional_impact = milestone[7]
            date = milestone[8].split('T')[0]  # Just the date part
            
            summary += f"Week {week}: {title}\n"
            summary += f"  💝 {description[:80]}...\n"
            summary += f"  💭 Emotional impact: {emotional_impact}\n"
            summary += f"  📅 {date}\n\n"
        
        summary += "🎉 Every milestone is a precious memory in your journey to motherhood!"
        
        return summary
    
    def suggest_milestone_tracking(self, current_week: int) -> List[str]:
        """Suggest milestones to watch for based on current pregnancy week"""
        suggestions = []
        
        for category, milestones in self.milestone_patterns.items():
            for milestone_name, patterns in milestones.items():
                if current_week in patterns["typical_weeks"]:
                    suggestion = f"Week {current_week}: Look out for {milestone_name.replace('_', ' ')}"
                    suggestions.append(suggestion)
        
        return suggestions