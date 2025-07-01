# =====================================
# File: core/SemanticEngagementCore.py
# Smart semantic understanding that handles real human communication
# =====================================
# or
#print("LOADED: THIS IS ROOT SemanticEngagementCore.py")

import re
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from difflib import SequenceMatcher
import numpy as np

class SemanticEngagementCore:
    """
    Advanced semantic understanding that works with:
    - Typos and misspellings
    - Emotional expressions
    - Incomplete thoughts
    - Stream of consciousness
    - Cultural variations
    """
    
    def __init__(self):
        # Semantic concept clusters (not rigid keywords)
        self.concept_clusters = {
            "pregnancy_experience": {
                "core_concepts": [
                    "carrying baby", "expecting child", "growing life", "maternal journey",
                    "pregnancy journey", "becoming mother", "baby development", "fetal growth",
                    "prenatal experience", "gestational period", "maternity process"
                ],
                "emotional_patterns": [
                    "excited about baby", "worried about pregnancy", "scared of birth",
                    "happy expecting", "anxious about baby", "nervous about delivery",
                    "overwhelmed with pregnancy", "joyful about motherhood"
                ],
                "experience_markers": [
                    "weeks along", "months pregnant", "due in", "trimester",
                    "feeling movement", "baby kicking", "growing belly", "doctor visits"
                ],
                "fuzzy_terms": [
                    "preggers", "preggo", "expecting", "bun in oven", "with child",
                    "knocked up", "babymama", "mama-to-be", "pregant", "pregnent"  # includes typos
                ],
                "domain": "pregnancy",
                "empathy_level": 0.8,
                "priority": 10
            },
            
            "emotional_distress": {
                "core_concepts": [
                    "feeling overwhelmed", "emotional struggle", "mental difficulty",
                    "psychological stress", "emotional burden", "inner turmoil",
                    "feeling lost", "emotional confusion", "mental exhaustion"
                ],
                "intensity_markers": [
                    "really struggling", "so overwhelmed", "can't handle", "falling apart",
                    "breaking down", "at my limit", "too much", "drowning in",
                    "completely lost", "don't know what to do"
                ],
                "vulnerability_signals": [
                    "scared to admit", "embarrassed to say", "don't want to bother",
                    "probably silly but", "might sound crazy", "hope you understand",
                    "hard to explain", "difficult to put into words"
                ],
                "fuzzy_emotions": [
                    "anxius", "anxyous", "woried", "depresed", "overwelmed",  # typos
                    "stresd", "confuzed", "scered", "nervus", "emoshunal"
                ],
                "domain": "emotional_support",
                "empathy_level": 0.9,
                "priority": 9
            },
            
            "physical_concerns": {
                "core_concepts": [
                    "body changes", "physical symptoms", "health concerns",
                    "bodily sensations", "medical worries", "physical discomfort"
                ],
                "urgency_patterns": [
                    "sudden pain", "severe symptoms", "emergency situation",
                    "urgent concern", "immediate help", "something wrong"
                ],
                "descriptive_language": [
                    "hurts so bad", "really painful", "strange feeling",
                    "never felt this", "something different", "not normal"
                ],
                "fuzzy_medical": [
                    "stomac ache", "hedache", "nausia", "bleading",  # typos
                    "craping", "sweling", "dizines", "weeknes"
                ],
                "domain": "health_support",
                "empathy_level": 0.7,
                "priority": 8
            },
            
            "seeking_connection": {
                "core_concepts": [
                    "need someone to talk", "feeling alone", "want understanding",
                    "need support", "looking for help", "seeking comfort"
                ],
                "isolation_markers": [
                    "no one understands", "feeling alone", "nobody gets it",
                    "isolated from others", "don't have anyone", "feel disconnected"
                ],
                "reaching_out": [
                    "hope you can help", "need someone who understands",
                    "looking for advice", "want to talk", "need guidance"
                ],
                "domain": "connection_support",
                "empathy_level": 0.85,
                "priority": 7
            }
        }
        
        # Initialize semantic analysis tools
        self._prepare_semantic_tools()
    
    def _prepare_semantic_tools(self):
        """Prepare tools for semantic analysis"""
        # Create combined vocabulary for fuzzy matching
        self.vocabulary = set()
        self.concept_map = {}
        
        for cluster_name, cluster_data in self.concept_clusters.items():
            for category in ['core_concepts', 'emotional_patterns', 'experience_markers', 'fuzzy_terms']:
                if category in cluster_data:
                    for phrase in cluster_data[category]:
                        words = phrase.lower().split()
                        for word in words:
                            self.vocabulary.add(word)
                            if word not in self.concept_map:
                                self.concept_map[word] = []
                            self.concept_map[word].append(cluster_name)
    
    def analyze_semantic_intent(self, text: str) -> Dict[str, Any]:
        """
        Deep semantic analysis that understands intent beyond keywords
        """
        # Clean and prepare text
        cleaned_text = self._normalize_text(text)
        
        # Multi-layer analysis
        emotional_state = self._analyze_emotional_state(cleaned_text)
        conceptual_matches = self._find_conceptual_matches(cleaned_text)
        communication_style = self._analyze_communication_style(cleaned_text)
        urgency_level = self._assess_urgency(cleaned_text)
        vulnerability_indicators = self._detect_vulnerability(cleaned_text)
        
        # Synthesize understanding
        primary_intent = self._synthesize_intent(
            emotional_state, conceptual_matches, communication_style, 
            urgency_level, vulnerability_indicators
        )
        
        return {
            "primary_intent": primary_intent,
            "emotional_state": emotional_state,
            "conceptual_matches": conceptual_matches,
            "communication_style": communication_style,
            "urgency_level": urgency_level,
            "vulnerability_indicators": vulnerability_indicators,
            "engagement_strategy": self._determine_engagement_strategy(primary_intent),
            "confidence_score": self._calculate_confidence(conceptual_matches),
            "timestamp": datetime.now().isoformat()
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text while preserving emotional markers"""
        # Fix common typos but keep emotional intensity
        text = text.lower()
        
        # Common pregnancy typos
        typo_fixes = {
            r'\bpregant\b': 'pregnant',
            r'\bpregnent\b': 'pregnant',
            r'\bpregnet\b': 'pregnant',
            r'\banxius\b': 'anxious',
            r'\banxyous\b': 'anxious',
            r'\bworied\b': 'worried',
            r'\bworryed\b': 'worried',
            r'\boverwelmed\b': 'overwhelmed',
            r'\boverwhelmd\b': 'overwhelmed',
            r'\bscered\b': 'scared',
            r'\bscared\b': 'scared',
            r'\bnervus\b': 'nervous',
            r'\bconfuzed\b': 'confused',
            r'\bdepresed\b': 'depressed',
            r'\btired\b': 'tired',
            r'\bexausted\b': 'exhausted'
        }
        
        for typo, correction in typo_fixes.items():
            text = re.sub(typo, correction, text)
        
        return text
    
    def _analyze_emotional_state(self, text: str) -> Dict[str, float]:
        """Analyze emotional undertones and intensity"""
        emotions = {
            "anxiety": 0.0,
            "joy": 0.0,
            "fear": 0.0,
            "sadness": 0.0,
            "excitement": 0.0,
            "overwhelm": 0.0,
            "confusion": 0.0,
            "vulnerability": 0.0
        }
        
        # Emotional intensity patterns
        intensity_patterns = {
            "anxiety": [
                r'anxious|worried|nervous|stress|tense',
                r'what if|concerned about|afraid that',
                r'can\'t stop thinking|keeps me up|on my mind'
            ],
            "overwhelm": [
                r'overwhelmed|too much|can\'t handle|drowning',
                r'so many|everything at once|falling apart',
                r'don\'t know how|impossible to'
            ],
            "vulnerability": [
                r'scared to|embarrassed|don\'t want to bother',
                r'probably silly|might sound|hope you understand',
                r'hard to explain|difficult to say'
            ],
            "joy": [
                r'excited|happy|thrilled|amazing|wonderful',
                r'can\'t wait|so grateful|blessed',
                r'best feeling|incredible|magical'
            ]
        }
        
        # Calculate emotional scores
        for emotion, patterns in intensity_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 0.3
            
            # Boost for repeated letters (emotional intensity)
            repeated_letters = len(re.findall(r'(\w)\1{2,}', text))
            score += repeated_letters * 0.1
            
            # Boost for capitalization (shouting/emphasis)
            caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
            score += caps_words * 0.2
            
            emotions[emotion] = min(score, 1.0)
        
        return emotions
    
    def _find_conceptual_matches(self, text: str) -> List[Dict[str, Any]]:
        """Find semantic matches using fuzzy logic"""
        matches = []
        
        for cluster_name, cluster_data in self.concept_clusters.items():
            cluster_score = 0.0
            matched_concepts = []
            
            # Check all concept categories
            for category in ['core_concepts', 'emotional_patterns', 'experience_markers', 'fuzzy_terms']:
                if category in cluster_data:
                    for concept in cluster_data[category]:
                        similarity = self._calculate_semantic_similarity(text, concept)
                        if similarity > 0.3:  # Threshold for relevance
                            cluster_score += similarity
                            matched_concepts.append({
                                "concept": concept,
                                "similarity": similarity,
                                "category": category
                            })
            
            if cluster_score > 0:
                matches.append({
                    "cluster": cluster_name,
                    "total_score": cluster_score,
                    "domain": cluster_data["domain"],
                    "empathy_level": cluster_data["empathy_level"],
                    "priority": cluster_data["priority"],
                    "matched_concepts": matched_concepts
                })
        
        # Sort by relevance
        matches.sort(key=lambda x: x["total_score"], reverse=True)
        return matches
    
    def _calculate_semantic_similarity(self, text: str, concept: str) -> float:
        """Calculate semantic similarity between text and concept"""
        # Multiple similarity measures
        
        # 1. Direct substring match
        if concept.lower() in text.lower():
            return 1.0
        
        # 2. Word overlap
        text_words = set(text.lower().split())
        concept_words = set(concept.lower().split())
        
        if concept_words:
            word_overlap = len(text_words.intersection(concept_words)) / len(concept_words)
            if word_overlap > 0.5:
                return word_overlap
        
        # 3. Fuzzy string matching for typos
        similarity = SequenceMatcher(None, text.lower(), concept.lower()).ratio()
        if similarity > 0.7:
            return similarity
        
        # 4. Individual word fuzzy matching
        max_word_similarity = 0.0
        for text_word in text_words:
            for concept_word in concept_words:
                word_sim = SequenceMatcher(None, text_word, concept_word).ratio()
                if word_sim > 0.8:  # High threshold for individual words
                    max_word_similarity = max(max_word_similarity, word_sim)
        
        return max_word_similarity
    
    def _analyze_communication_style(self, text: str) -> Dict[str, Any]:
        """Analyze how the person is communicating"""
        style = {
            "formality": "informal",
            "directness": "moderate",
            "emotional_openness": "moderate",
            "urgency_markers": 0,
            "uncertainty_markers": 0,
            "length": len(text.split())
        }
        
        # Detect communication patterns
        if re.search(r'please|thank you|could you|would you', text, re.IGNORECASE):
            style["formality"] = "formal"
        
        if re.search(r'!!|\.\.\.|\?!|help!', text):
            style["urgency_markers"] += 1
        
        if re.search(r'maybe|perhaps|i think|not sure|probably', text, re.IGNORECASE):
            style["uncertainty_markers"] += 1
        
        if re.search(r'feel|feeling|emotion|heart|soul', text, re.IGNORECASE):
            style["emotional_openness"] = "high"
        
        return style
    
    def _assess_urgency(self, text: str) -> Dict[str, Any]:
        """Assess urgency level from communication"""
        urgency_indicators = {
            "emergency_words": len(re.findall(r'emergency|urgent|help|now|immediately', text, re.IGNORECASE)),
            "pain_intensity": len(re.findall(r'severe|extreme|unbearable|intense', text, re.IGNORECASE)),
            "time_pressure": len(re.findall(r'right now|asap|quickly|fast', text, re.IGNORECASE)),
            "emotional_crisis": len(re.findall(r'breaking down|can\'t cope|falling apart', text, re.IGNORECASE))
        }
        
        total_urgency = sum(urgency_indicators.values())
        
        if total_urgency >= 3:
            level = "high"
        elif total_urgency >= 1:
            level = "moderate"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": total_urgency,
            "indicators": urgency_indicators
        }
    
    def _detect_vulnerability(self, text: str) -> Dict[str, Any]:
        """Detect vulnerability markers in communication"""
        vulnerability_signals = {
            "hesitation": len(re.findall(r'sorry to bother|hope this is ok|don\'t want to', text, re.IGNORECASE)),
            "shame": len(re.findall(r'embarrassed|ashamed|stupid|silly', text, re.IGNORECASE)),
            "isolation": len(re.findall(r'alone|no one|nobody|isolated', text, re.IGNORECASE)),
            "self_doubt": len(re.findall(r'probably wrong|might be silly|not sure if', text, re.IGNORECASE))
        }
        
        total_vulnerability = sum(vulnerability_signals.values())
        
        return {
            "level": "high" if total_vulnerability >= 2 else "moderate" if total_vulnerability >= 1 else "low",
            "signals": vulnerability_signals,
            "needs_reassurance": total_vulnerability >= 1
        }
    
    def _synthesize_intent(self, emotional_state, conceptual_matches, 
                          communication_style, urgency_level, vulnerability_indicators) -> Dict[str, Any]:
        """Synthesize all analysis into primary intent understanding"""
        
        if not conceptual_matches:
            # No clear conceptual match - analyze emotional state
            dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])
            
            return {
                "type": "emotional_expression",
                "primary_domain": "emotional_support",
                "confidence": 0.6,
                "dominant_emotion": dominant_emotion[0],
                "emotion_intensity": dominant_emotion[1],
                "needs_validation": True
            }
        
        # Use highest scoring conceptual match
        primary_match = conceptual_matches[0]
        
        return {
            "type": "domain_specific",
            "primary_domain": primary_match["domain"],
            "confidence": min(primary_match["total_score"], 1.0),
            "empathy_level": primary_match["empathy_level"],
            "priority": primary_match["priority"],
            "urgency": urgency_level["level"],
            "vulnerability": vulnerability_indicators["level"],
            "needs_reassurance": vulnerability_indicators["needs_reassurance"]
        }
    
    def _determine_engagement_strategy(self, primary_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how Friday should engage based on understanding"""
        
        strategy = {
            "tone": "warm",
            "approach": "supportive",
            "validation_level": "moderate",
            "information_depth": "moderate",
            "follow_up_questions": True
        }
        
        # Adjust based on intent
        if primary_intent.get("vulnerability") == "high":
            strategy.update({
                "tone": "gentle",
                "validation_level": "high",
                "approach": "reassuring"
            })
        
        if primary_intent.get("urgency") == "high":
            strategy.update({
                "approach": "direct",
                "information_depth": "high",
                "immediate_action": True
            })
        
        if primary_intent.get("primary_domain") == "pregnancy":
            strategy.update({
                "expertise_level": "high",
                "medical_awareness": True,
                "emotional_support": True
            })
        
        return strategy
    
    def _calculate_confidence(self, conceptual_matches: List[Dict]) -> float:
        """Calculate overall confidence in understanding"""
        if not conceptual_matches:
            return 0.3
        
        total_score = sum(match["total_score"] for match in conceptual_matches)
        normalized_score = min(total_score / 3.0, 1.0)  # Normalize to 0-1
        
        return normalized_score

# =====================================
# Enhanced Integration with FridayAI
# =====================================

class SemanticResponseEngine:
    """Generates contextually appropriate responses based on semantic understanding"""
    
    def __init__(self, semantic_core):
        self.semantic_core = semantic_core
    
    def generate_contextual_response(self, user_input: str, llm_response: str, 
                                   semantic_analysis: Dict[str, Any]) -> str:
        """Enhance LLM response with semantic understanding"""
        
        intent = semantic_analysis["primary_intent"]
        strategy = semantic_analysis["engagement_strategy"]
        
        # Build contextual prefix
        prefix = self._build_empathetic_prefix(intent, strategy)
        
        # Enhance response based on understanding
        enhanced_response = self._enhance_response_content(llm_response, intent, strategy)
        
        # Add supportive suffix if needed
        suffix = self._build_supportive_suffix(intent, strategy)
        
        return f"{prefix}{enhanced_response}{suffix}".strip()
    
    def _build_empathetic_prefix(self, intent: Dict, strategy: Dict) -> str:
        """Build empathetic opening based on understanding"""
        if strategy.get("validation_level") == "high":
            if intent.get("vulnerability") == "high":
                return "I can really hear how you're feeling right now, and I want you to know that what you're experiencing is completely valid. "
            elif intent.get("dominant_emotion") == "anxiety":
                return "I understand you're feeling anxious, and that's completely natural. "
        
        if intent.get("primary_domain") == "pregnancy":
            return "Thank you for sharing this with me. Pregnancy brings so many feelings and experiences. "
        
        return ""
    
    def _enhance_response_content(self, response: str, intent: Dict, strategy: Dict) -> str:
        """Enhance the main response content"""
        # Add medical awareness for health-related content
        if intent.get("primary_domain") == "health_support" and "medical" not in response.lower():
            response += " Please remember that I can provide support and information, but for any health concerns, it's always best to consult with your healthcare provider."
        
        return response
    
    def _build_supportive_suffix(self, intent: Dict, strategy: Dict) -> str:
        """Build supportive closing based on understanding"""
        suffixes = []
        
        if strategy.get("follow_up_questions") and intent.get("confidence", 0) > 0.7:
            if intent.get("primary_domain") == "pregnancy":
                suffixes.append(" Is there anything specific about your pregnancy journey you'd like to talk about?")
            elif intent.get("type") == "emotional_expression":
                suffixes.append(" How are you feeling right now? I'm here to listen.")
        
        if intent.get("needs_reassurance"):
            suffixes.append(" You're doing great by reaching out and taking care of yourself.")
        
        return "".join(suffixes)

# =====================================
# Replace your respond_to method with this enhanced version:
# =====================================

def enhanced_respond_to(self, user_input: str) -> Dict[str, object]:
    """Enhanced respond_to with semantic understanding"""
    
    # Deep semantic analysis
    semantic_analysis = self.semantic_engagement.analyze_semantic_intent(user_input)
    
   #print("\n[🧠 Semantic Analysis]")
    intent = semantic_analysis["primary_intent"]
    print(f"Intent: {intent['type']} | Domain: {intent.get('primary_domain', 'general')}")
    print(f"Confidence: {intent.get('confidence', 0):.2f} | Empathy Level: {intent.get('empathy_level', 0.5):.2f}")
    
    if semantic_analysis["urgency_level"]["level"] != "low":
        print(f"⚠️ Urgency: {semantic_analysis['urgency_level']['level']}")
    
    if semantic_analysis["vulnerability_indicators"]["needs_reassurance"]:
        print("💝 Needs reassurance detected")
    
    # Memory injection with semantic context
    ctx = inject(user_input)
    print(f"[🧠 Memory Injection] {ctx['reflection']}")
    
    # Enhanced knowledge query
    citations = query_knowledge(user_input)
    for c in citations:
        print(f"[📚 Quote from {c['source']}] {c['text']}")
    
    # Generate base response
    result = self.pipeline.generate_response(user_input)
    
    # Handle response format
    if isinstance(result, str):
        base_response = result
        emotional_tone = intent.get("dominant_emotion", "neutral")
    elif isinstance(result, dict):
        base_response = result.get('reply', result.get('response', '')).strip()
        emotional_tone = result.get('emotion', result.get('emotional_tone', 'neutral'))
    else:
        base_response = str(result)
        emotional_tone = "neutral"
    
    # Enhance response with semantic understanding
    final_response = self.semantic_response_engine.generate_contextual_response(
        user_input, base_response, semantic_analysis
    )
    
    # Format final output
    final_output = f"Friday: {final_response}"
    
    # Add citations
    if citations:
        sources = [f"📄 {c['source']}: {c['text']}" for c in citations if 'text' in c]
        final_output += "\n\n[🔍 Related Knowledge]\n" + "\n\n".join(sources)
    
    # Enhanced logging
    log_event(user_input, source="user", semantic_analysis=semantic_analysis)
    log_event(final_output, source="friday", intent=intent)
    
    # Update mood
    try:
        update_mood(emotional_tone)
    except Exception as e:
        print(f"[ERROR] update_mood failed: {e}")
        update_mood("neutral")
    
    return {
        "domain": intent.get("primary_domain", "general"),
        "content": final_output,
        "confidence": intent.get("confidence", 0.6),
        "emotional_tone": emotional_tone,
        "semantic_analysis": semantic_analysis,
        "engagement_strategy": semantic_analysis["engagement_strategy"],
        "processing_time": datetime.now().isoformat()
    }

# =====================================
# Add to your FridayAI.__init__ method:
# =====================================

def add_semantic_understanding(self):
    """Add this to your FridayAI.__init__ method"""
    self.semantic_engagement = SemanticEngagementCore()
    self.semantic_response_engine = SemanticResponseEngine(self.semantic_engagement)
    #print("[DEBUG] SemanticEngagementCore initialized")