# =====================================
# FILE: core/pregnancy/MaternalKnowledgeGraph.py
# Phase 1.3: Maternal Domain Knowledge Graph
# =====================================

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from ..GraphBrainCore import GraphBrainCore

@dataclass
class MedicalKnowledgeNode:
    """Represents medical knowledge with safety classifications"""
    node_id: str
    topic: str
    content: str
    safety_level: str  # 'safe', 'caution', 'medical_advice_needed', 'emergency'
    evidence_level: str  # 'strong', 'moderate', 'limited', 'anecdotal'
    trimester_relevance: List[int]  # [1, 2, 3]
    cultural_sensitivity: Dict[str, str]
    source_credibility: float  # 0.0 to 1.0
    last_updated: str
    tags: List[str]

@dataclass
class SafetyRecommendation:
    """Pregnancy-safe recommendations with risk assessment"""
    recommendation_id: str
    category: str
    recommendation: str
    safety_score: float  # 0.0 (dangerous) to 1.0 (completely safe)
    conditions: List[str]  # When this applies
    contraindications: List[str]  # When to avoid
    alternatives: List[str]  # Safe alternatives
    medical_consultation_required: bool

class MaternalKnowledgeGraph(GraphBrainCore):
    """
    Specialized knowledge graph for maternal care
    Builds on your existing GraphBrainCore with pregnancy-specific intelligence
    """
    
    def __init__(self, memory_core):
        super().__init__(memory_core)
        self.medical_knowledge = self._load_medical_knowledge()
        self.safety_protocols = self._load_safety_protocols()
        self.cultural_adaptations = self._load_cultural_adaptations()
        self._init_maternal_graph()
        
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load comprehensive pregnancy medical knowledge"""
        return {
            "nutrition": {
                "safe_foods": {
                    "proteins": {
                        "content": "Lean meats, fish low in mercury, eggs, legumes, tofu",
                        "safety_level": "safe",
                        "evidence_level": "strong",
                        "notes": "Ensure proper cooking temperatures"
                    },
                    "dairy": {
                        "content": "Pasteurized milk, cheese, yogurt",
                        "safety_level": "safe", 
                        "evidence_level": "strong",
                        "notes": "Avoid unpasteurized products"
                    },
                    "fruits_vegetables": {
                        "content": "Fresh, frozen, canned fruits and vegetables",
                        "safety_level": "safe",
                        "evidence_level": "strong",
                        "notes": "Wash thoroughly, avoid pre-cut items"
                    }
                },
                "foods_to_limit": {
                    "caffeine": {
                        "content": "Limit to 200mg per day (about 12oz coffee)",
                        "safety_level": "caution",
                        "evidence_level": "strong",
                        "reason": "High amounts linked to miscarriage risk"
                    },
                    "fish_high_mercury": {
                        "content": "Shark, swordfish, king mackerel, tilefish",
                        "safety_level": "medical_advice_needed",
                        "evidence_level": "strong",
                        "reason": "Mercury can harm baby's developing nervous system"
                    }
                },
                "foods_to_avoid": {
                    "raw_undercooked": {
                        "content": "Raw fish, undercooked meat, raw eggs",
                        "safety_level": "emergency",
                        "evidence_level": "strong",
                        "reason": "Risk of foodborne illness"
                    },
                    "unpasteurized": {
                        "content": "Unpasteurized dairy, soft cheeses",
                        "safety_level": "emergency",
                        "evidence_level": "strong",
                        "reason": "Risk of listeria infection"
                    }
                }
            },
            "exercise": {
                "safe_activities": {
                    "walking": {
                        "content": "30 minutes daily of brisk walking",
                        "safety_level": "safe",
                        "evidence_level": "strong",
                        "benefits": "Improves circulation, mood, energy"
                    },
                    "swimming": {
                        "content": "Low-impact water exercises",
                        "safety_level": "safe",
                        "evidence_level": "strong",
                        "benefits": "Full body workout, supports joints"
                    },
                    "prenatal_yoga": {
                        "content": "Modified yoga poses for pregnancy",
                        "safety_level": "safe",
                        "evidence_level": "moderate",
                        "benefits": "Flexibility, stress relief, preparation for birth"
                    }
                },
                "activities_to_modify": {
                    "weight_lifting": {
                        "content": "Light weights, avoid lying flat after 20 weeks",
                        "safety_level": "caution",
                        "evidence_level": "moderate",
                        "modifications": "Seated or standing positions only"
                    }
                },
                "activities_to_avoid": {
                    "contact_sports": {
                        "content": "Football, hockey, martial arts",
                        "safety_level": "emergency",
                        "evidence_level": "strong",
                        "reason": "Risk of abdominal trauma"
                    },
                    "high_altitude": {
                        "content": "Activities above 6000 feet if not accustomed",
                        "safety_level": "medical_advice_needed",
                        "evidence_level": "moderate",
                        "reason": "Reduced oxygen availability"
                    }
                }
            },
            "medications": {
                "generally_safe": {
                    "acetaminophen": {
                        "content": "Tylenol for pain and fever",
                        "safety_level": "safe",
                        "evidence_level": "strong",
                        "dosage": "As directed on package"
                    },
                    "prenatal_vitamins": {
                        "content": "Vitamins specifically formulated for pregnancy",
                        "safety_level": "safe",
                        "evidence_level": "strong",
                        "importance": "Provides essential nutrients"
                    }
                },
                "use_with_caution": {
                    "antihistamines": {
                        "content": "Some allergy medications",
                        "safety_level": "medical_advice_needed",
                        "evidence_level": "moderate",
                        "note": "Consult healthcare provider first"
                    }
                },
                "avoid": {
                    "nsaids": {
                        "content": "Ibuprofen, aspirin, naproxen",
                        "safety_level": "emergency",
                        "evidence_level": "strong",
                        "reason": "Can affect baby's heart and kidney development"
                    },
                    "retinoids": {
                        "content": "Acne medications containing retinoids",
                        "safety_level": "emergency",
                        "evidence_level": "strong",
                        "reason": "Can cause birth defects"
                    }
                }
            },
            "warning_signs": {
                "emergency": {
                    "severe_bleeding": {
                        "content": "Heavy bleeding with clots",
                        "safety_level": "emergency",
                        "action": "Call emergency services immediately"
                    },
                    "severe_abdominal_pain": {
                        "content": "Intense cramping or pain",
                        "safety_level": "emergency",
                        "action": "Seek immediate medical attention"
                    },
                    "signs_of_preeclampsia": {
                        "content": "Severe headache, vision changes, upper belly pain",
                        "safety_level": "emergency",
                        "action": "Contact healthcare provider immediately"
                    }
                },
                "concerning": {
                    "decreased_fetal_movement": {
                        "content": "Significant reduction in baby's movement",
                        "safety_level": "medical_advice_needed",
                        "action": "Contact healthcare provider same day"
                    },
                    "persistent_vomiting": {
                        "content": "Unable to keep food or fluids down",
                        "safety_level": "medical_advice_needed",
                        "action": "Contact healthcare provider"
                    }
                }
            }
        }
    
    def _load_safety_protocols(self) -> Dict[str, Any]:
        """Load safety assessment protocols"""
        return {
            "risk_assessment_matrix": {
                "trimester_1": {
                    "high_risk_activities": ["hot_tubs", "saunas", "x_rays", "certain_medications"],
                    "moderate_risk": ["caffeine_excess", "stress", "poor_nutrition"],
                    "monitoring_focus": ["bleeding", "cramping", "nausea_severity"]
                },
                "trimester_2": {
                    "high_risk_activities": ["contact_sports", "activities_with_fall_risk"],
                    "moderate_risk": ["back_sleeping", "heavy_lifting"],
                    "monitoring_focus": ["fetal_movement", "blood_pressure", "weight_gain"]
                },
                "trimester_3": {
                    "high_risk_activities": ["lying_on_back", "high_intensity_exercise"],
                    "moderate_risk": ["travel", "standing_long_periods"],
                    "monitoring_focus": ["contractions", "swelling", "baby_position"]
                }
            },
            "emergency_protocols": {
                "immediate_911": [
                    "severe_bleeding", "severe_abdominal_pain", "signs_of_stroke",
                    "difficulty_breathing", "chest_pain", "thoughts_of_self_harm"
                ],
                "call_healthcare_provider": [
                    "decreased_fetal_movement", "persistent_headache", "vision_changes",
                    "unusual_discharge", "contractions_before_37_weeks"
                ],
                "monitor_at_home": [
                    "mild_cramping", "braxton_hicks", "round_ligament_pain",
                    "mild_swelling", "normal_pregnancy_discomforts"
                ]
            }
        }
    
    def _load_cultural_adaptations(self) -> Dict[str, Any]:
        """Load cultural sensitivity adaptations"""
        return {
            "dietary_considerations": {
                "vegetarian": {
                    "protein_sources": ["legumes", "quinoa", "nuts", "dairy"],
                    "supplements": ["B12", "iron", "omega_3"],
                    "special_considerations": "Ensure adequate protein combining"
                },
                "vegan": {
                    "protein_sources": ["legumes", "quinoa", "nuts", "seeds"],
                    "supplements": ["B12", "iron", "omega_3", "calcium", "vitamin_D"],
                    "special_considerations": "Monitor B12 levels closely"
                },
                "religious_dietary_laws": {
                    "halal": "Ensure meat is halal certified",
                    "kosher": "Follow kosher preparation guidelines",
                    "hindu": "Respect vegetarian preferences and food timing"
                }
            },
            "birth_practices": {
                "western": {
                    "common_practices": ["hospital_birth", "pain_medication_options"],
                    "support_people": ["partner", "doula", "family"]
                },
                "traditional": {
                    "practices": ["home_birth", "traditional_midwifery", "herbal_support"],
                    "postpartum": ["confinement_period", "specific_foods", "family_support"]
                }
            },
            "communication_styles": {
                "direct": "Clear, straightforward medical information",
                "indirect": "Gentle, contextual guidance with family involvement",
                "high_context": "Consider family dynamics and cultural expectations"
            }
        }
    
    def _init_maternal_graph(self):
        """Initialize maternal-specific knowledge graph structure"""
        
        # Create core maternal care nodes
        self.add_node("maternal_care", "root", {"type": "domain_root"})
        
        # Add major category nodes
        major_categories = ["nutrition", "exercise", "medications", "mental_health", 
                           "medical_care", "birth_preparation", "postpartum"]
        
        for category in major_categories:
            self.add_node(category, "category", {"domain": "maternal_care"})
            self.add_edge("maternal_care", category, "contains")
        
        # Load detailed knowledge into graph
        self._populate_knowledge_graph()
    
    def _populate_knowledge_graph(self):
        """Populate graph with detailed medical knowledge"""
        
        for main_category, subcategories in self.medical_knowledge.items():
            for sub_category, items in subcategories.items():
                
                # Create subcategory node
                sub_node_id = f"{main_category}_{sub_category}"
                self.add_node(sub_node_id, "subcategory", {
                    "parent_category": main_category,
                    "type": "knowledge_category"
                })
                self.add_edge(main_category, sub_node_id, "contains")
                
                # Add individual knowledge items
                for item_name, item_data in items.items():
                    item_node_id = f"{sub_node_id}_{item_name}"
                    
                    node_attributes = {
                        "content": item_data.get("content", ""),
                        "safety_level": item_data.get("safety_level", "safe"),
                        "evidence_level": item_data.get("evidence_level", "moderate"),
                        "type": "knowledge_item"
                    }
                    
                    self.add_node(item_node_id, "knowledge_item", node_attributes)
                    self.add_edge(sub_node_id, item_node_id, "contains")
    
    def get_safe_recommendations(self, query: str, pregnancy_week: int = 0, 
                               user_context: Dict = None) -> List[SafetyRecommendation]:
        """Get pregnancy-safe recommendations for a query"""
        
        # Determine trimester
        trimester = self._calculate_trimester(pregnancy_week)
        
        # Search knowledge graph
        relevant_nodes = self.search_knowledge(query, pregnancy_week, user_context)
        
        recommendations = []
        for node in relevant_nodes:
            if node.get("safety_level") in ["safe", "caution"]:
                recommendation = SafetyRecommendation(
                    recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category=node.get("parent_category", "general"),
                    recommendation=node.get("content", ""),
                    safety_score=self._calculate_safety_score(node, trimester),
                    conditions=self._extract_conditions(node),
                    contraindications=self._extract_contraindications(node),
                    alternatives=self._find_alternatives(node),
                    medical_consultation_required=node.get("safety_level") == "medical_advice_needed"
                )
                recommendations.append(recommendation)
        
        # Sort by safety score
        recommendations.sort(key=lambda x: x.safety_score, reverse=True)
        
        return recommendations
    
    def search_knowledge(self, query: str, pregnancy_week: int = 0, 
                        user_context: Dict = None) -> List[Dict]:
        """Search maternal knowledge with safety filtering"""
        
        # Convert query to search terms
        search_terms = query.lower().split()
        
        # Search through knowledge graph
        relevant_nodes = []
        
        for node_id, node_data in self.nodes.items():
            if node_data.get("type") == "knowledge_item":
                
                # Check content relevance
                content = node_data.get("content", "").lower()
                relevance_score = self._calculate_relevance(search_terms, content)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    
                    # Add safety context
                    enhanced_node = node_data.copy()
                    enhanced_node["relevance_score"] = relevance_score
                    enhanced_node["pregnancy_safety"] = self._assess_pregnancy_safety(
                        node_data, pregnancy_week
                    )
                    
                    relevant_nodes.append(enhanced_node)
        
        # Sort by relevance and safety
        relevant_nodes.sort(key=lambda x: (x["relevance_score"], 
                                         x["pregnancy_safety"]["safety_score"]), 
                           reverse=True)
        
        return relevant_nodes[:10]  # Return top 10 results
    
    def _calculate_relevance(self, search_terms: List[str], content: str) -> float:
        """Calculate relevance score between search terms and content"""
        matches = sum(1 for term in search_terms if term in content)
        return matches / len(search_terms) if search_terms else 0.0
    
    def _assess_pregnancy_safety(self, node: Dict, pregnancy_week: int) -> Dict:
        """Assess pregnancy safety of a knowledge item"""
        
        safety_level = node.get("safety_level", "safe")
        trimester = self._calculate_trimester(pregnancy_week)
        
        # Base safety scores
        safety_scores = {
            "safe": 1.0,
            "caution": 0.7,
            "medical_advice_needed": 0.4,
            "emergency": 0.0
        }
        
        base_score = safety_scores.get(safety_level, 0.5)
        
        # Adjust for trimester-specific risks
        trimester_adjustment = self._get_trimester_adjustment(node, trimester)
        
        final_score = max(0.0, min(1.0, base_score + trimester_adjustment))
        
        return {
            "safety_score": final_score,
            "safety_level": safety_level,
            "trimester_specific_notes": self._get_trimester_notes(node, trimester),
            "requires_medical_consultation": safety_level == "medical_advice_needed"
        }
    
    def _calculate_trimester(self, pregnancy_week: int) -> int:
        """Calculate trimester from pregnancy week"""
        if pregnancy_week <= 12:
            return 1
        elif pregnancy_week <= 27:
            return 2
        else:
            return 3
    
    def _get_trimester_adjustment(self, node: Dict, trimester: int) -> float:
        """Get trimester-specific safety adjustments"""
        
        category = node.get("parent_category", "")
        
        # Trimester-specific adjustments
        adjustments = {
            1: {  # First trimester - more cautious
                "medications": -0.2,
                "exercise": -0.1,
                "nutrition": 0.0
            },
            2: {  # Second trimester - generally safest
                "medications": 0.0,
                "exercise": 0.1,
                "nutrition": 0.0
            },
            3: {  # Third trimester - position and activity restrictions
                "medications": -0.1,
                "exercise": -0.2,
                "nutrition": 0.0
            }
        }
        
        return adjustments.get(trimester, {}).get(category, 0.0)
    
    def _get_trimester_notes(self, node: Dict, trimester: int) -> str:
        """Get trimester-specific notes for recommendations"""
        
        category = node.get("parent_category", "")
        content = node.get("content", "")
        
        if trimester == 1 and "exercise" in category:
            return "First trimester: Listen to your body, avoid overheating"
        elif trimester == 2 and "exercise" in category:
            return "Second trimester: Great time to be active, avoid lying flat"
        elif trimester == 3 and "exercise" in category:
            return "Third trimester: Modify activities, avoid supine positions"
        
        return ""
    
    def _calculate_safety_score(self, node: Dict, trimester: int) -> float:
        """Calculate overall safety score"""
        return self._assess_pregnancy_safety(node, trimester * 13)["safety_score"]
    
    def _extract_conditions(self, node: Dict) -> List[str]:
        """Extract conditions when recommendation applies"""
        # This would parse conditions from node content
        return ["general_pregnancy"]
    
    def _extract_contraindications(self, node: Dict) -> List[str]:
        """Extract contraindications"""
        safety_level = node.get("safety_level", "safe")
        if safety_level == "emergency":
            return ["all_pregnancy_stages"]
        return []
    
    def _find_alternatives(self, node: Dict) -> List[str]:
        """Find safe alternatives"""
        # This would search for alternative recommendations
        return ["consult_healthcare_provider"]
    
    def generate_personalized_guidance(self, user_query: str, user_context: Dict) -> Dict[str, Any]:
        """Generate personalized, culturally-sensitive guidance"""
        
        pregnancy_week = user_context.get("pregnancy_week", 0)
        cultural_background = user_context.get("cultural_background", "western")
        medical_history = user_context.get("medical_history", [])
        dietary_preferences = user_context.get("dietary_preferences", [])
        
        # Get safe recommendations
        recommendations = self.get_safe_recommendations(user_query, pregnancy_week, user_context)
        
        # Apply cultural adaptations
        culturally_adapted = self._apply_cultural_adaptations(
            recommendations, cultural_background, dietary_preferences
        )
        
        # Generate response
        guidance = {
            "primary_recommendations": culturally_adapted[:3],
            "safety_notes": self._generate_safety_notes(recommendations),
            "cultural_considerations": self._get_cultural_considerations(
                cultural_background, user_query
            ),
            "when_to_consult_doctor": self._get_consultation_triggers(recommendations),
            "evidence_level": self._assess_evidence_quality(recommendations),
            "confidence_score": self._calculate_guidance_confidence(recommendations, user_context)
        }
        
        return guidance
    
    def _apply_cultural_adaptations(self, recommendations: List[SafetyRecommendation], 
                                   cultural_background: str, dietary_preferences: List[str]) -> List[SafetyRecommendation]:
        """Apply cultural adaptations to recommendations"""
        
        adapted_recommendations = []
        
        for rec in recommendations:
            adapted_rec = rec
            
            # Apply dietary adaptations
            if rec.category == "nutrition":
                if "vegetarian" in dietary_preferences:
                    adapted_rec = self._adapt_for_vegetarian(rec)
                elif "vegan" in dietary_preferences:
                    adapted_rec = self._adapt_for_vegan(rec)
                elif "halal" in dietary_preferences:
                    adapted_rec = self._adapt_for_halal(rec)
            
            # Apply cultural communication style
            if cultural_background in self.cultural_adaptations["communication_styles"]:
                style = self.cultural_adaptations["communication_styles"][cultural_background]
                adapted_rec.recommendation = self._adapt_communication_style(
                    adapted_rec.recommendation, style
                )
            
            adapted_recommendations.append(adapted_rec)
        
        return adapted_recommendations
    
    def _adapt_for_vegetarian(self, recommendation: SafetyRecommendation) -> SafetyRecommendation:
        """Adapt nutrition recommendations for vegetarian diet"""
        if "meat" in recommendation.recommendation.lower():
            veg_alternatives = self.cultural_adaptations["dietary_considerations"]["vegetarian"]
            alternative_proteins = ", ".join(veg_alternatives["protein_sources"])
            recommendation.recommendation = recommendation.recommendation.replace(
                "lean meats", f"vegetarian proteins like {alternative_proteins}"
            )
            recommendation.alternatives.extend(veg_alternatives["protein_sources"])
        return recommendation
    
    def _adapt_for_vegan(self, recommendation: SafetyRecommendation) -> SafetyRecommendation:
        """Adapt nutrition recommendations for vegan diet"""
        if any(animal_product in recommendation.recommendation.lower() 
               for animal_product in ["meat", "dairy", "eggs", "fish"]):
            vegan_alternatives = self.cultural_adaptations["dietary_considerations"]["vegan"]
            recommendation.recommendation += f" For vegan alternatives: {', '.join(vegan_alternatives['protein_sources'])}"
            recommendation.alternatives.extend(vegan_alternatives["protein_sources"])
            
            # Add supplement recommendations
            supplements = vegan_alternatives["supplements"]
            recommendation.recommendation += f" Important supplements: {', '.join(supplements)}"
        return recommendation
    
    def _adapt_for_halal(self, recommendation: SafetyRecommendation) -> SafetyRecommendation:
        """Adapt recommendations for halal dietary requirements"""
        if "meat" in recommendation.recommendation.lower():
            recommendation.recommendation = recommendation.recommendation.replace(
                "lean meats", "halal-certified lean meats"
            )
        return recommendation
    
    def _adapt_communication_style(self, text: str, style: str) -> str:
        """Adapt communication style based on cultural preferences"""
        if style == "direct":
            return text  # Keep as is for direct communication
        elif style == "indirect":
            return f"You might consider {text.lower()}" if not text.startswith("You might") else text
        elif style == "high_context":
            return f"Many mothers in your situation find that {text.lower()}"
        return text
    
    def _generate_safety_notes(self, recommendations: List[SafetyRecommendation]) -> List[str]:
        """Generate important safety notes"""
        safety_notes = []
        
        # Check for any high-risk recommendations
        high_risk = [r for r in recommendations if r.safety_score < 0.6]
        if high_risk:
            safety_notes.append("⚠️ Some recommendations require medical consultation")
        
        # Check for emergency-level items
        emergency_items = [r for r in recommendations if r.medical_consultation_required]
        if emergency_items:
            safety_notes.append("🏥 Please discuss with your healthcare provider before proceeding")
        
        # Add general pregnancy safety reminder
        safety_notes.append("💝 Every pregnancy is unique - trust your body and instincts")
        
        return safety_notes
    
    def _get_cultural_considerations(self, cultural_background: str, query: str) -> List[str]:
        """Get cultural considerations for the query"""
        considerations = []
        
        if cultural_background in self.cultural_adaptations.get("birth_practices", {}):
            practices = self.cultural_adaptations["birth_practices"][cultural_background]
            if "birth" in query.lower() or "delivery" in query.lower():
                considerations.append(f"Consider {practices.get('common_practices', ['traditional practices'])[0]}")
        
        if "food" in query.lower() or "eat" in query.lower():
            considerations.append("Dietary recommendations respect your cultural food traditions")
        
        return considerations
    
    def _get_consultation_triggers(self, recommendations: List[SafetyRecommendation]) -> List[str]:
        """Determine when medical consultation is needed"""
        triggers = []
        
        consultation_needed = [r for r in recommendations if r.medical_consultation_required]
        if consultation_needed:
            triggers.append("Before implementing any recommendations marked as requiring consultation")
        
        triggers.extend([
            "If you experience any unusual symptoms",
            "If you have concerns about your specific health conditions",
            "Before making significant changes to diet or exercise"
        ])
        
        return triggers
    
    def _assess_evidence_quality(self, recommendations: List[SafetyRecommendation]) -> str:
        """Assess overall evidence quality of recommendations"""
        if not recommendations:
            return "limited"
        
        # This would analyze the evidence levels of source knowledge
        # For now, return a general assessment
        return "strong"  # Based on medical consensus
    
    def _calculate_guidance_confidence(self, recommendations: List[SafetyRecommendation], 
                                     user_context: Dict) -> float:
        """Calculate confidence in the guidance provided"""
        base_confidence = 0.7
        
        # Increase confidence if we have user context
        if user_context.get("pregnancy_week", 0) > 0:
            base_confidence += 0.1
        
        # Increase confidence if recommendations are high safety
        if recommendations:
            avg_safety = sum(r.safety_score for r in recommendations) / len(recommendations)
            if avg_safety > 0.8:
                base_confidence += 0.1
        
        # Decrease confidence if medical consultation required
        if any(r.medical_consultation_required for r in recommendations):
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def get_emergency_protocols(self, symptoms: List[str]) -> Dict[str, Any]:
        """Get emergency response protocols for reported symptoms"""
        
        emergency_level = "monitor"  # Default
        immediate_actions = []
        
        # Check against emergency protocols
        emergency_symptoms = self.safety_protocols["emergency_protocols"]["immediate_911"]
        urgent_symptoms = self.safety_protocols["emergency_protocols"]["call_healthcare_provider"]
        
        for symptom in symptoms:
            if any(emergency in symptom.lower() for emergency in emergency_symptoms):
                emergency_level = "emergency"
                immediate_actions.append("Call emergency services (911) immediately")
                break
            elif any(urgent in symptom.lower() for urgent in urgent_symptoms):
                emergency_level = "urgent"
                immediate_actions.append("Contact your healthcare provider today")
        
        return {
            "emergency_level": emergency_level,
            "immediate_actions": immediate_actions,
            "symptoms_to_monitor": self._get_monitoring_guidance(symptoms),
            "when_to_escalate": self._get_escalation_criteria(),
            "support_resources": self._get_support_resources(emergency_level)
        }
    
    def _get_monitoring_guidance(self, symptoms: List[str]) -> List[str]:
        """Get guidance on monitoring symptoms"""
        return [
            "Track frequency and severity of symptoms",
            "Note any patterns or triggers",
            "Keep a record to share with healthcare provider"
        ]
    
    def _get_escalation_criteria(self) -> List[str]:
        """Get criteria for when to escalate care"""
        return [
            "Symptoms worsen or become more frequent",
            "New concerning symptoms develop",
            "You feel something is seriously wrong"
        ]
    
    def _get_support_resources(self, emergency_level: str) -> List[str]:
        """Get appropriate support resources"""
        if emergency_level == "emergency":
            return ["Emergency services: 911", "Hospital emergency department"]
        elif emergency_level == "urgent":
            return ["Healthcare provider", "Nurse hotline", "Urgent care clinic"]
        else:
            return ["Healthcare provider", "Pregnancy support groups", "Trusted family/friends"]
    
    def generate_knowledge_summary(self, topic: str, pregnancy_week: int = 0) -> str:
        """Generate a comprehensive knowledge summary for a topic"""
        
        # Search for all relevant knowledge
        relevant_nodes = self.search_knowledge(topic, pregnancy_week)
        
        if not relevant_nodes:
            return f"I don't have specific information about {topic} in my pregnancy knowledge base. Please consult your healthcare provider."
        
        summary = f"📚 **{topic.title()} During Pregnancy**\n\n"
        
        # Group by safety level
        safe_items = [n for n in relevant_nodes if n.get("safety_level") == "safe"]
        caution_items = [n for n in relevant_nodes if n.get("safety_level") == "caution"]
        avoid_items = [n for n in relevant_nodes if n.get("safety_level") in ["medical_advice_needed", "emergency"]]
        
        if safe_items:
            summary += "✅ **Generally Safe:**\n"
            for item in safe_items[:3]:
                summary += f"• {item.get('content', '')}\n"
            summary += "\n"
        
        if caution_items:
            summary += "⚠️ **Use with Caution:**\n"
            for item in caution_items[:3]:
                summary += f"• {item.get('content', '')}\n"
            summary += "\n"
        
        if avoid_items:
            summary += "❌ **Avoid or Consult Doctor:**\n"
            for item in avoid_items[:3]:
                summary += f"• {item.get('content', '')}\n"
            summary += "\n"
        
        summary += "💝 **Remember:** Every pregnancy is unique. Always discuss with your healthcare provider for personalized advice."
        
        return summary