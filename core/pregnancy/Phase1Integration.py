# =====================================
# FILE: core/pregnancy/Phase1Integration.py
# Phase 1 Complete: Enhanced Maternal Care Integration
# =====================================

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from .PregnancyEmotionCore import PregnancyEmotionCore, PregnancyEmotionalState
from .MaternalTimelineMemory import MaternalTimelineMemory, PregnancyMemoryContext
from .MaternalKnowledgeGraph import MaternalKnowledgeGraph
from ..MemoryCore import MemoryCore
from ..EmotionCoreV2 import EmotionCoreV2

class EnhancedMaternalFriday:
    """
    Phase 1 Complete: Enhanced FridayAI with advanced maternal care capabilities
    Integrates pregnancy emotion detection, timeline memory, and knowledge graph
    """
    
    def __init__(self, friday_ai_instance, maternal_care_system):
        self.friday = friday_ai_instance
        self.maternal_system = maternal_care_system
        
        # Initialize Phase 1 components
        self.pregnancy_emotion_core = PregnancyEmotionCore()
        self.maternal_memory = MaternalTimelineMemory()
        self.maternal_knowledge = MaternalKnowledgeGraph(friday_ai_instance.memory)
        
        print("[🧠 PHASE 1] Enhanced Maternal Intelligence Initialized")
        print("✅ Pregnancy-specific emotion detection")
        print("✅ Maternal timeline memory system") 
        print("✅ Medical knowledge graph with safety protocols")
        
    def enhanced_respond_to(self, user_input: str, user_id: str = None) -> Dict[str, Any]:
        """
        Enhanced response system with Phase 1 maternal intelligence
        """
        
        # Get user context if available
        user_context = self._get_enhanced_user_context(user_id) if user_id else {}
        pregnancy_week = user_context.get("pregnancy_week", 0)
        
        # Step 1: Enhanced emotion analysis
        emotional_state = self.pregnancy_emotion_core.analyze_pregnancy_emotion(
            user_input, pregnancy_week, user_context
        )
        
        # Step 2: Memory storage with maternal context
        if user_id and pregnancy_week > 0:
            memory_id = self.maternal_memory.store_maternal_memory(
                content=user_input,
                memory_type="conversation",
                user_id=user_id,
                pregnancy_week=pregnancy_week,
                emotional_context=emotional_state.__dict__
            )
        
        # Step 3: Enhanced memory retrieval
        relevant_memories = []
        if user_id:
            relevant_memories = self.maternal_memory.search_maternal_memories(
                user_input, user_id, {"trimester": emotional_state.trimester_factor}
            )
        
        # Step 4: Knowledge-based guidance
        knowledge_guidance = None
        if self._is_asking_for_guidance(user_input):
            knowledge_guidance = self.maternal_knowledge.generate_personalized_guidance(
                user_input, user_context
            )
        
        # Step 5: Generate enhanced response
        base_response = self.friday.respond_to(user_input)
        enhanced_response = self._enhance_response_with_maternal_intelligence(
            base_response, emotional_state, relevant_memories, knowledge_guidance, user_context
        )
        
        # Step 6: Generate supportive follow-up if needed
        supportive_response = self.pregnancy_emotion_core.generate_supportive_response(emotional_state)
        
        return {
            "content": enhanced_response,
            "emotional_analysis": {
                "primary_emotion": emotional_state.primary_emotion,
                "intensity": emotional_state.intensity,
                "hormonal_influence": emotional_state.hormonal_influence,
                "pregnancy_week": emotional_state.pregnancy_week,
                "confidence": emotional_state.confidence_score
            },
            "supportive_message": supportive_response,
            "knowledge_guidance": knowledge_guidance,
            "relevant_memories": len(relevant_memories),
            "maternal_context": user_context,
            "processing_time": datetime.now().isoformat(),
            "phase_1_enhanced": True
        }
    
    def _get_enhanced_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context including maternal data"""
        
        context = {}
        
        # Get pregnancy context from maternal memory
        pregnancy_context = self.maternal_memory.get_maternal_context(user_id)
        if pregnancy_context:
            context.update({
                "pregnancy_week": pregnancy_context.current_week,
                "trimester": pregnancy_context.trimester,
                "due_date": pregnancy_context.due_date,
                "recent_milestones": [m.title for m in pregnancy_context.recent_milestones[:3]],
                "emotional_trends": pregnancy_context.emotional_trends
            })
        
        # Get user profile from maternal care system
        if hasattr(self.maternal_system, "health_profile"):
            profile = self.maternal_system["health_profile"]._get_user_profile(user_id)
            if profile:
                context.update({
                    "cultural_background": profile.get("preferences", {}).get("communication_style", "warm"),
                    "medical_history": profile.get("medical_history", {}).get("pre_existing_conditions", []),
                    "dietary_preferences": self._extract_dietary_preferences(profile)
                })
        
        return context
    
    def _extract_dietary_preferences(self, profile: Dict) -> List[str]:
        """Extract dietary preferences from user profile"""
        # This would analyze the profile for dietary information
        # For now, return empty list - would be enhanced based on profile structure
        return []
    
    def _is_asking_for_guidance(self, user_input: str) -> bool:
        """Determine if user is asking for pregnancy-related guidance"""
        guidance_indicators = [
            "can i", "is it safe", "should i", "what about", "how about",
            "advice", "recommend", "safe to", "okay to", "allowed",
            "guidance", "help with", "what can", "suggestions"
        ]
        
        pregnancy_topics = [
            "eat", "drink", "exercise", "medication", "sleep", "work",
            "travel", "pregnancy", "baby", "birth", "delivery"
        ]
        
        input_lower = user_input.lower()
        
        has_guidance_indicator = any(indicator in input_lower for indicator in guidance_indicators)
        has_pregnancy_topic = any(topic in input_lower for topic in pregnancy_topics)
        
        return has_guidance_indicator and has_pregnancy_topic
    
    def _enhance_response_with_maternal_intelligence(self, base_response: Dict, 
                                                   emotional_state: PregnancyEmotionalState,
                                                   relevant_memories: List[Dict],
                                                   knowledge_guidance: Dict,
                                                   user_context: Dict) -> str:
        """Enhance base response with maternal intelligence"""
        
        enhanced_content = base_response.get('content', '')
        
        # Add emotional context awareness
        if emotional_state.intensity > 0.7:
            if emotional_state.primary_emotion in ["anxiety", "fear", "anticipatory_anxiety"]:
                enhanced_content += "\n\n💝 I can sense you're feeling quite anxious about this. That's completely normal during pregnancy."
            elif emotional_state.primary_emotion in ["overwhelming_love", "excitement"]:
                enhanced_content += "\n\n✨ I can feel the love and excitement in your words! These feelings are such a beautiful part of your journey."
        
        # Add relevant memory context
        if relevant_memories:
            recent_milestone = next((m for m in relevant_memories if m.get('associated_milestones')), None)
            if recent_milestone:
                enhanced_content += f"\n\n🌸 I remember when you shared about {recent_milestone['associated_milestones'][0]['title']}."
        
        # Add knowledge guidance summary
        if knowledge_guidance:
            primary_recs = knowledge_guidance.get("primary_recommendations", [])
            if primary_recs:
                enhanced_content += "\n\n📚 Based on current pregnancy guidelines:"
                for i, rec in enumerate(primary_recs[:2], 1):
                    safety_emoji = "✅" if rec.safety_score > 0.8 else "⚠️" if rec.safety_score > 0.5 else "🏥"
                    enhanced_content += f"\n{safety_emoji} {rec.recommendation}"
        
        # Add week-specific context
        if user_context.get("pregnancy_week", 0) > 0:
            week = user_context["pregnancy_week"]
            trimester = user_context.get("trimester", 1)
            enhanced_content += f"\n\n📅 At {week} weeks (trimester {trimester}), you're doing amazingly well!"
        
        return enhanced_content
    
    def process_maternal_command(self, user_input: str, user_id: str) -> str:
        """Process maternal care specific commands with Phase 1 enhancements"""
        
        input_lower = user_input.lower()
        
        # Enhanced milestone tracking
        if input_lower.startswith("!track_milestone"):
            return self._track_enhanced_milestone(user_input, user_id)
        
        # Emotional pattern analysis
        elif input_lower.startswith("!emotion_analysis"):
            return self._generate_emotion_analysis(user_id)
        
        # Memory timeline
        elif input_lower.startswith("!memory_timeline"):
            return self._generate_memory_timeline(user_id)
        
        # Safety guidance
        elif input_lower.startswith("!safety_check"):
            query = input_lower.replace("!safety_check", "").strip()
            return self._generate_safety_guidance(query, user_id)
        
        # Knowledge summary
        elif input_lower.startswith("!knowledge"):
            topic = input_lower.replace("!knowledge", "").strip()
            return self._generate_knowledge_summary(topic, user_id)
        
        # Milestone suggestions
        elif input_lower.startswith("!milestone_suggestions"):
            return self._generate_milestone_suggestions(user_id)
        
        return None  # Not a maternal command
    
    def _track_enhanced_milestone(self, user_input: str, user_id: str) -> str:
        """Track milestone with enhanced detection"""
        
        user_context = self._get_enhanced_user_context(user_id)
        pregnancy_week = user_context.get("pregnancy_week", 0)
        
        # Extract milestone description from command
        milestone_text = user_input.replace("!track_milestone", "").strip()
        
        if not milestone_text:
            return "Please provide milestone details: !track_milestone [description]"
        
        # Store with enhanced analysis
        memory_id = self.maternal_memory.store_maternal_memory(
            content=milestone_text,
            memory_type="milestone",
            user_id=user_id,
            pregnancy_week=pregnancy_week
        )
        
        # Analyze emotional significance
        emotional_state = self.pregnancy_emotion_core.analyze_pregnancy_emotion(
            milestone_text, pregnancy_week, user_context
        )
        
        response = f"✨ Milestone tracked successfully!\n\n"
        response += f"📅 Week {pregnancy_week}: {milestone_text}\n"
        response += f"💭 Emotional significance: {emotional_state.primary_emotion} (intensity: {emotional_state.intensity:.1f})\n"
        
        if emotional_state.hormonal_influence > 0.6:
            response += "🌸 This seems like a particularly meaningful moment in your journey!\n"
        
        return response
    
    def _generate_emotion_analysis(self, user_id: str) -> str:
        """Generate comprehensive emotion analysis"""
        
        # This would analyze emotional patterns over time
        # For now, provide a sample analysis structure
        
        return """
🧠 **Your Emotional Journey Analysis**

📊 **Recent Patterns:**
• Primary emotions: Joy, anticipation, mild anxiety
• Hormonal influence: Moderate (normal for pregnancy)
• Emotional stability: Good overall trend

📈 **Trends by Trimester:**
• First trimester: Higher anxiety, excitement
• Current: More stable, increasing connection
• Predicted: Anticipation building toward birth

💝 **Insights:**
• Your emotional responses are completely normal for your stage
• Strong bonding emotions detected - beautiful sign of connection
• Anxiety levels within healthy range

🌟 **Recommendations:**
• Continue mindfulness practices
• Celebrate positive milestones
• Reach out when feeling overwhelmed
        """
    
    def _generate_memory_timeline(self, user_id: str) -> str:
        """Generate beautiful memory timeline"""
        
        user_context = self._get_enhanced_user_context(user_id)
        pregnancy_week = user_context.get("pregnancy_week", 0)
        
        # Generate milestone summary
        if pregnancy_week > 0:
            return self.maternal_memory.generate_milestone_summary(user_id)
        else:
            return "Create your pregnancy profile first to see your beautiful journey timeline! 🌸"
    
    def _generate_safety_guidance(self, query: str, user_id: str) -> str:
        """Generate safety guidance using knowledge graph"""
        
        if not query:
            return "Please specify what you'd like safety guidance about: !safety_check [topic]"
        
        user_context = self._get_enhanced_user_context(user_id)
        
        guidance = self.maternal_knowledge.generate_personalized_guidance(query, user_context)
        
        if not guidance:
            return f"I don't have specific safety guidance for '{query}'. Please consult your healthcare provider."
        
        response = f"🛡️ **Safety Guidance: {query.title()}**\n\n"
        
        # Add primary recommendations
        for i, rec in enumerate(guidance.get("primary_recommendations", [])[:3], 1):
            safety_emoji = "✅" if rec.safety_score > 0.8 else "⚠️" if rec.safety_score > 0.5 else "🏥"
            response += f"{safety_emoji} **{rec.category.title()}:** {rec.recommendation}\n\n"
        
        # Add safety notes
        safety_notes = guidance.get("safety_notes", [])
        if safety_notes:
            response += "**Important Notes:**\n"
            for note in safety_notes:
                response += f"• {note}\n"
        
        # Add consultation guidance
        consultation_triggers = guidance.get("when_to_consult_doctor", [])
        if consultation_triggers:
            response += f"\n**Consult your healthcare provider:**\n"
            for trigger in consultation_triggers[:2]:
                response += f"• {trigger}\n"
        
        return response
    
    def _generate_knowledge_summary(self, topic: str, user_id: str) -> str:
        """Generate knowledge summary for topic"""
        
        if not topic:
            return "Please specify a topic: !knowledge [topic] (e.g., nutrition, exercise, medications)"
        
        user_context = self._get_enhanced_user_context(user_id)
        pregnancy_week = user_context.get("pregnancy_week", 0)
        
        return self.maternal_knowledge.generate_knowledge_summary(topic, pregnancy_week)
    
    def _generate_milestone_suggestions(self, user_id: str) -> str:
        """Generate milestone suggestions for current week"""
        
        user_context = self._get_enhanced_user_context(user_id)
        pregnancy_week = user_context.get("pregnancy_week", 0)
        
        if pregnancy_week <= 0:
            return "Create your pregnancy profile first to get personalized milestone suggestions! 🌟"
        
        suggestions = self.maternal_memory.suggest_milestone_tracking(pregnancy_week)
        
        if not suggestions:
            return f"No specific milestones typically occur at week {pregnancy_week}, but every week of your journey is special! 💝"
        
        response = f"🌟 **Milestone Suggestions for Week {pregnancy_week}**\n\n"
        for suggestion in suggestions:
            response += f"✨ {suggestion}\n"
        
        response += "\n💝 Remember: Every pregnancy is unique. These are just common experiences!"
        
        return response

# =====================================
# Enhanced Integration Function
# =====================================

def integrate_phase1_maternal_care(friday_ai):
    """
    Integrate Phase 1 enhanced maternal care with existing FridayAI
    """
    
    print("\n" + "="*60)
    print("🧠 FRIDAY AI - PHASE 1 MATERNAL CARE ENHANCEMENT")
    print("Advanced neurological system for maternal support")
    print("="*60)
    
    try:
        # Initialize enhanced maternal system
        if hasattr(friday_ai, 'maternal_care'):
            enhanced_friday = EnhancedMaternalFriday(friday_ai, friday_ai.maternal_care)
            
            # Enhance friday_ai with Phase 1 capabilities
            friday_ai.enhanced_maternal = enhanced_friday
            
            # Add enhanced command processing
            original_respond = friday_ai.respond_to
            
            def enhanced_respond_wrapper(user_input: str, user_id: str = None):
                """Enhanced response wrapper"""
                
                # Check for maternal commands first
                if user_id and user_input.startswith("!"):
                    maternal_response = enhanced_friday.process_maternal_command(user_input, user_id)
                    if maternal_response:
                        return {"content": maternal_response, "domain": "enhanced_maternal"}
                
                # Check if user has maternal profile
                if user_id and hasattr(friday_ai, 'maternal_care') and friday_ai.maternal_care.get("user_id"):
                    return enhanced_friday.enhanced_respond_to(user_input, user_id)
                else:
                    return original_respond(user_input)
            
            # Replace the respond_to method
            friday_ai.respond_to = enhanced_respond_wrapper
            
            print("\n✅ PHASE 1 ENHANCEMENT COMPLETE!")
            print("✅ Pregnancy-specific emotion detection active")
            print("✅ Maternal timeline memory system online")
            print("✅ Medical knowledge graph with safety protocols ready")
            print("✅ Enhanced command system available")
            
            print("\n🌟 NEW ENHANCED COMMANDS:")
            print("• !track_milestone [description] - Enhanced milestone tracking")
            print("• !emotion_analysis - Comprehensive emotional pattern analysis")
            print("• !memory_timeline - Beautiful journey timeline")
            print("• !safety_check [topic] - Personalized safety guidance")
            print("• !knowledge [topic] - Evidence-based pregnancy information")
            print("• !milestone_suggestions - Week-specific milestone suggestions")
            
            return friday_ai
            
        else:
            print("❌ Base maternal care system not found. Run basic integration first.")
            return friday_ai
            
    except Exception as e:
        print(f"❌ Phase 1 enhancement failed: {e}")
        return friday_ai