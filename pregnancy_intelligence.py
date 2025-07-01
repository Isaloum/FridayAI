# =====================================
# File: pregnancy_intelligence.py
# Purpose: Complete Pregnancy Intelligence System - Emotional Analysis, Support, and Medical Knowledge
# Phase 1, Step 3 of Brain Modularization
# =====================================

import random
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class PregnancyStage:
    """Comprehensive pregnancy stage information"""
    week: int
    trimester: int
    stage_name: str
    physical_changes: List[str]
    emotional_changes: List[str]
    baby_development: List[str]
    common_concerns: List[str]
    support_tips: List[str]
    medical_milestones: List[str]

class PregnancyKnowledge:
    """Advanced pregnancy knowledge base with comprehensive information"""
    
    def __init__(self):
        self.stages = self._init_pregnancy_stages()
        self.symptoms_guide = self._init_symptoms_guide()
        self.emotional_support = self._init_emotional_support()
        self.nutrition_guide = self._init_nutrition_guide()
        self.exercise_guide = self._init_exercise_guide()
        self.warning_signs = self._init_warning_signs()
        
    def _init_pregnancy_stages(self) -> Dict[int, PregnancyStage]:
        """Initialize comprehensive pregnancy stage information"""
        stages = {}
        
        # First Trimester (Weeks 1-12)
        first_trimester_data = {
            4: PregnancyStage(
                week=4, trimester=1, stage_name="Early Discovery",
                physical_changes=["Missed period", "Breast tenderness", "Mild cramping"],
                emotional_changes=["Excitement", "Anxiety", "Disbelief"],
                baby_development=["Neural tube forming", "Heart begins to develop"],
                common_concerns=["Is this real?", "When to tell people?", "Morning sickness starting?"],
                support_tips=["Take prenatal vitamins", "Avoid alcohol and smoking", "Rest when needed"],
                medical_milestones=["Confirm pregnancy", "Calculate due date", "First prenatal appointment"]
            ),
            6: PregnancyStage(
                week=6, trimester=1, stage_name="Morning Sickness Peak",
                physical_changes=["Nausea", "Fatigue", "Food aversions", "Frequent urination"],
                emotional_changes=["Mood swings", "Worry about symptoms", "Anticipation"],
                baby_development=["Heart beating", "Brain developing", "Limb buds forming"],
                common_concerns=["How long will nausea last?", "Is this normal?", "Diet worries"],
                support_tips=["Eat small frequent meals", "Stay hydrated", "Rest plenty"],
                medical_milestones=["First prenatal visit", "Blood work", "Medical history review"]
            ),
            8: PregnancyStage(
                week=8, trimester=1, stage_name="Embryonic Development",
                physical_changes=["Continued nausea", "Breast changes", "Bloating"],
                emotional_changes=["Adjustment to pregnancy", "Relationship changes", "Future planning"],
                baby_development=["Major organs forming", "Facial features developing", "Movement beginning"],
                common_concerns=["Miscarriage risk", "Work adjustments", "Partner involvement"],
                support_tips=["Communicate with partner", "Take it easy", "Follow medical advice"],
                medical_milestones=["Possible first ultrasound", "Confirm heartbeat", "Discuss nutrition"]
            ),
            12: PregnancyStage(
                week=12, trimester=1, stage_name="First Trimester End",
                physical_changes=["Nausea may decrease", "Energy returning", "Visible body changes"],
                emotional_changes=["Relief", "Excitement to share news", "Confidence growing"],
                baby_development=["All major organs formed", "Reflexes developing", "Growth accelerating"],
                common_concerns=["When to announce?", "Maternity clothes?", "Work planning"],
                support_tips=["Share your news when ready", "Plan for second trimester", "Celebrate milestone"],
                medical_milestones=["NT scan option", "Genetic testing discussion", "Routine checkup"]
            )
        }
        
        # Second Trimester (Weeks 13-27)
        second_trimester_data = {
            16: PregnancyStage(
                week=16, trimester=2, stage_name="Golden Period Beginning",
                physical_changes=["Energy returning", "Glowing skin", "Growing belly"],
                emotional_changes=["Increased confidence", "Bonding with baby", "Planning excitement"],
                baby_development=["Sex determinable", "Hair growing", "Stronger movements"],
                common_concerns=["Feeling movements", "Baby's health", "Body image"],
                support_tips=["Enjoy increased energy", "Start prenatal classes", "Bond with baby"],
                medical_milestones=["Possible anatomy scan", "Alpha-fetoprotein test", "Regular checkups"]
            ),
            20: PregnancyStage(
                week=20, trimester=2, stage_name="Halfway Point",
                physical_changes=["Obvious belly", "Feeling movements", "Weight gain"],
                emotional_changes=["Reality setting in", "Excitement", "Some anxiety"],
                baby_development=["Fully formed", "Active movements", "Hearing developing"],
                common_concerns=["Normal weight gain?", "Baby's movements", "Preparing nursery"],
                support_tips=["Track baby movements", "Plan nursery", "Take photos"],
                medical_milestones=["Anatomy scan", "Gender reveal option", "Growth assessment"]
            ),
            24: PregnancyStage(
                week=24, trimester=2, stage_name="Viability Milestone",
                physical_changes=["Stronger kicks", "Back pain", "Stretch marks possible"],
                emotional_changes=["Deep bonding", "Future focus", "Some worries"],
                baby_development=["Viability reached", "Rapid brain development", "Sleep cycles"],
                common_concerns=["Preterm labor", "Baby's position", "Birth planning"],
                support_tips=["Learn about preterm signs", "Practice relaxation", "Stay active"],
                medical_milestones=["Glucose screening", "Viability milestone", "Growth monitoring"]
            )
        }
        
        # Third Trimester (Weeks 28-40+)
        third_trimester_data = {
            28: PregnancyStage(
                week=28, trimester=3, stage_name="Third Trimester Beginning",
                physical_changes=["Larger belly", "Shortness of breath", "Frequent urination returns"],
                emotional_changes=["Anticipation", "Anxiety about labor", "Nesting instinct"],
                baby_development=["Eyes opening", "Fat accumulating", "Brain maturing"],
                common_concerns=["Labor preparation", "Baby's position", "Work leave"],
                support_tips=["Start birth classes", "Practice breathing", "Prepare birth plan"],
                medical_milestones=["More frequent visits", "Group B strep test later", "Growth monitoring"]
            ),
            32: PregnancyStage(
                week=32, trimester=3, stage_name="Preparation Phase",
                physical_changes=["Significant discomfort", "Braxton Hicks", "Sleep difficulties"],
                emotional_changes=["Eager to meet baby", "Labor anxiety", "Nesting intensity"],
                baby_development=["Rapid weight gain", "Lung development", "Positioning for birth"],
                common_concerns=["Early labor signs", "Hospital bag", "Pain management"],
                support_tips=["Pack hospital bag", "Tour hospital", "Practice labor positions"],
                medical_milestones=["Weekly visits soon", "Fetal monitoring", "Birth plan discussion"]
            ),
            36: PregnancyStage(
                week=36, trimester=3, stage_name="Full Term Approaching",
                physical_changes=["Maximum discomfort", "Lightning possible", "Strong movements"],
                emotional_changes=["Ready to be done", "Excitement", "Final preparations"],
                baby_development=["Nearly full term", "Final organ maturation", "Head positioning"],
                common_concerns=["Labor timing", "Final preparations", "Support person availability"],
                support_tips=["Stay alert for labor signs", "Rest when possible", "Have support ready"],
                medical_milestones=["Weekly visits", "Cervical checks", "Labor discussion"]
            ),
            40: PregnancyStage(
                week=40, trimester=3, stage_name="Due Date",
                physical_changes=["Maximum size", "Possible early labor", "Anticipation"],
                emotional_changes=["Excitement", "Impatience", "Labor anticipation"],
                baby_development=["Full term", "Ready for birth", "Final preparations"],
                common_concerns=["When will labor start?", "Overdue concerns", "Induction possibility"],
                support_tips=["Stay calm", "Walk when comfortable", "Trust your body"],
                medical_milestones=["Due date assessment", "Possible induction discussion", "Final preparations"]
            )
        }
        
        # Combine all stages
        stages.update(first_trimester_data)
        stages.update(second_trimester_data)
        stages.update(third_trimester_data)
        
        return stages
    
    def _init_symptoms_guide(self) -> Dict[str, Dict]:
        """Initialize comprehensive symptoms guide"""
        return {
            'nausea': {
                'description': 'Morning sickness - nausea and vomiting',
                'when': 'Usually weeks 6-12, but can vary',
                'normal_range': 'Mild to moderate nausea, occasional vomiting',
                'relief_tips': [
                    'Eat small, frequent meals',
                    'Try ginger tea or ginger supplements',
                    'Avoid triggers (smells, foods)',
                    'Stay hydrated with small sips',
                    'Rest when possible',
                    'Consider B6 supplements (consult doctor)'
                ],
                'when_to_worry': 'Severe vomiting, dehydration, weight loss, inability to keep fluids down'
            },
            'fatigue': {
                'description': 'Extreme tiredness and low energy',
                'when': 'First trimester and third trimester',
                'normal_range': 'Feeling more tired than usual, needing more sleep',
                'relief_tips': [
                    'Get plenty of sleep (8-9 hours)',
                    'Take short naps when possible',
                    'Eat iron-rich foods',
                    'Stay hydrated',
                    'Light exercise when energy permits',
                    'Ask for help with daily tasks'
                ],
                'when_to_worry': 'Extreme exhaustion, fainting, inability to function'
            },
            'back_pain': {
                'description': 'Lower back pain and discomfort',
                'when': 'Second and third trimester',
                'normal_range': 'Mild to moderate lower back ache',
                'relief_tips': [
                    'Practice good posture',
                    'Wear supportive shoes',
                    'Use pregnancy support belt',
                    'Sleep on your side with pillow support',
                    'Gentle prenatal yoga',
                    'Warm baths (not too hot)'
                ],
                'when_to_worry': 'Severe pain, radiating pain, fever with pain'
            },
            'heartburn': {
                'description': 'Burning sensation in chest/throat',
                'when': 'Second and third trimester',
                'normal_range': 'Occasional to frequent burning sensation',
                'relief_tips': [
                    'Eat smaller, more frequent meals',
                    'Avoid spicy, greasy, or acidic foods',
                    'Stay upright after eating',
                    'Sleep with head elevated',
                    'Chew gum to increase saliva',
                    'Consult doctor about safe antacids'
                ],
                'when_to_worry': 'Severe pain, difficulty swallowing, persistent symptoms'
            }
        }
    
    def _init_emotional_support(self) -> Dict[str, Dict]:
        """Initialize emotional support guidance"""
        return {
            'anxiety': {
                'description': 'Worry, fear, or nervousness about pregnancy or future',
                'common_triggers': [
                    'Baby\'s health concerns',
                    'Labor and delivery fears',
                    'Financial worries',
                    'Body changes',
                    'Relationship changes',
                    'Parenting readiness'
                ],
                'coping_strategies': [
                    'Practice deep breathing exercises',
                    'Talk to trusted friends or family',
                    'Join pregnancy support groups',
                    'Consider prenatal counseling',
                    'Practice mindfulness or meditation',
                    'Focus on what you can control',
                    'Limit exposure to negative stories',
                    'Create a support network'
                ],
                'professional_help': 'Consider professional help if anxiety interferes with daily life, sleep, or eating'
            },
            'mood_swings': {
                'description': 'Rapid changes in emotional state',
                'common_triggers': [
                    'Hormonal changes',
                    'Physical discomfort',
                    'Stress and fatigue',
                    'Life changes',
                    'Relationship adjustments'
                ],
                'coping_strategies': [
                    'Acknowledge that mood swings are normal',
                    'Communicate with your partner',
                    'Get adequate rest',
                    'Maintain regular eating schedule',
                    'Practice stress-reduction techniques',
                    'Keep a mood journal',
                    'Ask for patience from loved ones'
                ],
                'professional_help': 'Seek help if mood swings are severe or accompanied by thoughts of self-harm'
            },
            'body_image': {
                'description': 'Concerns about changing body and appearance',
                'common_concerns': [
                    'Weight gain',
                    'Changing shape',
                    'Skin changes',
                    'Feeling unattractive',
                    'Clothes not fitting',
                    'Partner attraction'
                ],
                'positive_strategies': [
                    'Focus on what your body is accomplishing',
                    'Practice self-compassion',
                    'Invest in comfortable, flattering maternity clothes',
                    'Take photos to document your journey',
                    'Talk to other mothers about their experiences',
                    'Communicate with your partner about your feelings',
                    'Practice body-positive self-talk'
                ],
                'professional_help': 'Consider counseling if body image concerns are causing significant distress'
            }
        }
    
    def _init_nutrition_guide(self) -> Dict[str, Any]:
        """Initialize comprehensive nutrition guidance"""
        return {
            'essential_nutrients': {
                'folate': {
                    'importance': 'Prevents neural tube defects',
                    'amount': '600-800 mcg daily',
                    'sources': ['Leafy greens', 'Fortified cereals', 'Beans', 'Citrus fruits']
                },
                'iron': {
                    'importance': 'Prevents anemia, supports increased blood volume',
                    'amount': '27 mg daily',
                    'sources': ['Lean meat', 'Spinach', 'Beans', 'Fortified cereals']
                },
                'calcium': {
                    'importance': 'Baby\'s bone development, maintains your bone health',
                    'amount': '1000 mg daily',
                    'sources': ['Dairy products', 'Fortified plant milks', 'Leafy greens', 'Sardines']
                },
                'protein': {
                    'importance': 'Baby\'s growth and development',
                    'amount': '70-100g daily',
                    'sources': ['Lean meat', 'Fish', 'Eggs', 'Beans', 'Nuts', 'Dairy']
                },
                'omega3': {
                    'importance': 'Brain and eye development',
                    'amount': '200-300 mg DHA daily',
                    'sources': ['Fatty fish', 'Walnuts', 'Flaxseeds', 'Supplements']
                }
            },
            'foods_to_avoid': [
                'Raw or undercooked meat, fish, eggs',
                'High-mercury fish (shark, swordfish, king mackerel)',
                'Unpasteurized dairy products',
                'Raw sprouts',
                'Unwashed fruits and vegetables',
                'Excessive caffeine (limit to 200mg/day)',
                'Alcohol',
                'High-sodium processed foods'
            ],
            'meal_planning_tips': [
                'Eat small, frequent meals',
                'Include protein with each meal',
                'Choose whole grains over refined',
                'Aim for colorful fruits and vegetables',
                'Stay hydrated (8-10 glasses water daily)',
                'Take prenatal vitamins as directed',
                'Listen to your body\'s hunger cues'
            ]
        }
    
    def _init_exercise_guide(self) -> Dict[str, Any]:
        """Initialize safe exercise guidance"""
        return {
            'safe_exercises': {
                'walking': {
                    'benefits': ['Cardiovascular health', 'Easy on joints', 'Mood improvement'],
                    'tips': ['Start slowly', 'Wear supportive shoes', 'Stay hydrated']
                },
                'swimming': {
                    'benefits': ['Full body workout', 'Joint support', 'Reduces swelling'],
                    'tips': ['Avoid hot tubs', 'Use pool with good hygiene', 'Don\'t overexert']
                },
                'prenatal_yoga': {
                    'benefits': ['Flexibility', 'Stress relief', 'Birth preparation'],
                    'tips': ['Avoid deep twists', 'No hot yoga', 'Listen to your body']
                },
                'strength_training': {
                    'benefits': ['Maintains muscle', 'Supports posture', 'Easier recovery'],
                    'tips': ['Use lighter weights', 'Avoid supine position after first trimester', 'Focus on form']
                }
            },
            'exercises_to_avoid': [
                'Contact sports',
                'Activities with fall risk',
                'Scuba diving',
                'Hot yoga',
                'Heavy weightlifting',
                'Exercises lying on back after first trimester'
            ],
            'general_guidelines': [
                'Aim for 150 minutes moderate exercise weekly',
                'Stop if you feel dizzy, short of breath, or have pain',
                'Stay hydrated and avoid overheating',
                'Consult your healthcare provider before starting new exercise',
                'Listen to your body and modify as needed'
            ]
        }
    
    def _init_warning_signs(self) -> Dict[str, Dict]:
        """Initialize warning signs that require medical attention"""
        return {
            'emergency_signs': {
                'description': 'Signs requiring immediate medical attention',
                'signs': [
                    'Heavy bleeding',
                    'Severe abdominal pain',
                    'Signs of preeclampsia (severe headache, vision changes, upper abdominal pain)',
                    'Decreased fetal movement after 28 weeks',
                    'Leaking amniotic fluid',
                    'Signs of preterm labor before 37 weeks',
                    'Persistent vomiting with inability to keep fluids down',
                    'High fever (over 101°F)',
                    'Severe shortness of breath',
                    'Chest pain'
                ],
                'action': 'Call emergency services or go to emergency room immediately'
            },
            'concerning_signs': {
                'description': 'Signs requiring prompt medical consultation',
                'signs': [
                    'Persistent severe headaches',
                    'Sudden swelling of face, hands, or feet',
                    'Changes in vision',
                    'Burning sensation during urination',
                    'Unusual vaginal discharge',
                    'Persistent back pain',
                    'Regular contractions before 37 weeks',
                    'Sudden decrease in fetal movement',
                    'Severe mood changes or depression'
                ],
                'action': 'Contact your healthcare provider within 24 hours'
            }
        }
    
    def get_stage_info(self, week: int) -> Optional[PregnancyStage]:
        """Get pregnancy stage information for specific week"""
        # Find the closest week with data
        available_weeks = sorted(self.stages.keys())
        closest_week = min(available_weeks, key=lambda x: abs(x - week))
        
        if abs(closest_week - week) <= 2:  # Within 2 weeks
            return self.stages[closest_week]
        return None
    
    def get_trimester_info(self, week: int) -> Dict[str, Any]:
        """Get trimester-specific information"""
        if week <= 12:
            trimester = 1
            focus = "Early development, managing symptoms, establishing care"
        elif week <= 27:
            trimester = 2
            focus = "Growth monitoring, enjoying energy, preparing for baby"
        else:
            trimester = 3
            focus = "Final preparations, labor readiness, baby positioning"
        
        return {
            'trimester': trimester,
            'focus': focus,
            'weeks_range': f"Weeks {(trimester-1)*13 + 1}-{min(trimester*13 + (1 if trimester < 3 else 7), 40)}"
        }
    
    def analyze_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Analyze reported symptoms and provide guidance"""
        analysis = {
            'normal_symptoms': [],
            'concerning_symptoms': [],
            'relief_tips': [],
            'medical_attention': False
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            
            # Check against known symptoms
            for known_symptom, info in self.symptoms_guide.items():
                if known_symptom in symptom_lower:
                    analysis['normal_symptoms'].append(known_symptom)
                    analysis['relief_tips'].extend(info['relief_tips'])
            
            # Check for concerning symptoms
            for category, warning_info in self.warning_signs.items():
                for warning_sign in warning_info['signs']:
                    if any(word in symptom_lower for word in warning_sign.lower().split()):
                        analysis['concerning_symptoms'].append(symptom)
                        analysis['medical_attention'] = True
        
        return analysis
    
    def get_emotional_support(self, emotion: str) -> Dict[str, Any]:
        """Get emotional support based on reported emotion"""
        emotion_lower = emotion.lower()
        
        for support_type, info in self.emotional_support.items():
            if support_type in emotion_lower or any(trigger.lower() in emotion_lower for trigger in info.get('common_triggers', [])):
                return {
                    'type': support_type,
                    'description': info['description'],
                    'strategies': info['coping_strategies'],
                    'professional_help': info['professional_help']
                }
        
        # Default emotional support
        return {
            'type': 'general',
            'description': 'General emotional support during pregnancy',
            'strategies': [
                'Talk to trusted friends or family',
                'Practice self-care and relaxation',
                'Consider joining pregnancy support groups',
                'Focus on positive aspects of pregnancy',
                'Remember that emotional changes are normal'
            ],
            'professional_help': 'Consider counseling if emotions feel overwhelming'
        }
    
    def get_nutrition_advice(self, concern: str = "") -> Dict[str, Any]:
        """Get nutrition advice based on specific concern or general guidance"""
        if not concern:
            return {
                'general_advice': 'Focus on balanced nutrition with essential pregnancy nutrients',
                'key_nutrients': list(self.nutrition_guide['essential_nutrients'].keys()),
                'meal_tips': self.nutrition_guide['meal_planning_tips'][:5]
            }
        
        concern_lower = concern.lower()
        advice = {'specific_advice': []}
        
        # Address specific nutritional concerns
        if any(word in concern_lower for word in ['nausea', 'sick', 'vomit']):
            advice['specific_advice'] = [
                'Try bland, dry foods like crackers',
                'Eat small, frequent meals',
                'Ginger may help with nausea',
                'Stay hydrated with small sips'
            ]
        elif any(word in concern_lower for word in ['weight', 'gain', 'fat']):
            advice['specific_advice'] = [
                'Focus on nutrient quality over quantity',
                'Aim for gradual, steady weight gain',
                'Include protein with each meal',
                'Choose whole foods over processed'
            ]
        elif any(word in concern_lower for word in ['energy', 'tired', 'fatigue']):
            advice['specific_advice'] = [
                'Eat iron-rich foods',
                'Include complex carbohydrates',
                'Stay hydrated',
                'Don\'t skip meals'
            ]
        
        advice['key_nutrients'] = self.nutrition_guide['essential_nutrients']
        return advice
    
    def get_exercise_guidance(self, current_activity: str = "", concerns: str = "") -> Dict[str, Any]:
        """Get exercise guidance based on current activity level and concerns"""
        guidance = {
            'safe_options': [],
            'modifications': [],
            'warnings': self.exercise_guide['exercises_to_avoid'],
            'general_tips': self.exercise_guide['general_guidelines']
        }
        
        # Recommend safe exercises
        for exercise, info in self.exercise_guide['safe_exercises'].items():
            guidance['safe_options'].append({
                'activity': exercise,
                'benefits': info['benefits'],
                'tips': info['tips']
            })
        
        # Address specific concerns
        if concerns:
            concerns_lower = concerns.lower()
            if any(word in concerns_lower for word in ['back', 'pain', 'ache']):
                guidance['modifications'].append('Focus on posture-supporting exercises like swimming and prenatal yoga')
            if any(word in concerns_lower for word in ['tired', 'energy', 'fatigue']):
                guidance['modifications'].append('Start with gentle activities like walking and build gradually')
            if any(word in concerns_lower for word in ['balance', 'dizzy']):
                guidance['modifications'].append('Avoid activities with balance requirements, focus on seated or supported exercises')
        
        return guidance


class PregnancyIntelligence:
    """Main pregnancy intelligence system that integrates all pregnancy knowledge"""
        
    def __init__(self, memory=None, emotion=None, identity=None):
        self.memory = memory
        self.emotion = emotion
        self.identity = identity

        self.knowledge = PregnancyKnowledge()
        self.conversation_context = {}
        self.user_profiles = {}
        
    def analyze_pregnancy_context(self, user_input: str, user_id: str = "default", pregnancy_week: int = 0) -> Dict[str, Any]:
        """Analyze user input for pregnancy-related context and needs"""
        input_lower = user_input.lower()
        
        analysis = {
            'pregnancy_related': False,
            'emotional_tone': 'neutral',
            'concerns': [],
            'topics': [],
            'support_needed': [],
            'medical_attention': False,
            'week_specific': False
        }
        
        # Check if pregnancy-related
        pregnancy_keywords = [
            'baby', 'pregnant', 'pregnancy', 'trimester', 'due date', 'labor', 'delivery',
            'morning sickness', 'ultrasound', 'prenatal', 'maternity', 'contractions',
            'kicks', 'movement', 'heartburn', 'nausea', 'fatigue', 'craving'
        ]
        
        if any(keyword in input_lower for keyword in pregnancy_keywords):
            analysis['pregnancy_related'] = True
        
        # Emotional tone analysis
        emotional_indicators = {
            'anxious': ['worried', 'scared', 'anxious', 'nervous', 'afraid', 'panic'],
            'excited': ['excited', 'happy', 'thrilled', 'can\'t wait', 'amazing'],
            'confused': ['confused', 'don\'t understand', 'unclear', 'mixed up'],
            'sad': ['sad', 'depressed', 'down', 'crying', 'upset'],
            'overwhelmed': ['overwhelmed', 'too much', 'can\'t handle', 'stressed']
        }
        
        for emotion, indicators in emotional_indicators.items():
            if any(indicator in input_lower for indicator in indicators):
                analysis['emotional_tone'] = emotion
                break
        
        # Topic identification
        topic_keywords = {
            'symptoms': ['nausea', 'tired', 'sick', 'pain', 'ache', 'symptoms'],
            'nutrition': ['eat', 'food', 'diet', 'nutrition', 'vitamin', 'craving'],
            'exercise': ['exercise', 'workout', 'yoga', 'walk', 'active', 'fitness'],
            'medical': ['doctor', 'appointment', 'test', 'ultrasound', 'checkup'],
            'emotional': ['feel', 'emotion', 'mood', 'stress', 'anxiety', 'worry'],
            'baby_development': ['baby', 'growth', 'development', 'movement', 'kicks'],
            'labor_delivery': ['labor', 'delivery', 'birth', 'contractions', 'due date']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                analysis['topics'].append(topic)
        
        # Support needs identification
        if analysis['emotional_tone'] in ['anxious', 'overwhelmed', 'sad']:
            analysis['support_needed'].append('emotional_support')
        
        if any(topic in analysis['topics'] for topic in ['symptoms', 'medical']):
            analysis['support_needed'].append('medical_guidance')
        
        if 'nutrition' in analysis['topics']:
            analysis['support_needed'].append('nutrition_advice')
        
        if 'exercise' in analysis['topics']:
            analysis['support_needed'].append('exercise_guidance')
        
        # Check for warning signs
        warning_keywords = [
            'bleeding', 'severe pain', 'can\'t breathe', 'dizzy', 'faint',
            'baby not moving', 'water broke', 'contractions'
        ]
        
        if any(warning in input_lower for warning in warning_keywords):
            analysis['medical_attention'] = True
        
        # Week-specific analysis
        if pregnancy_week > 0:
            analysis['week_specific'] = True
            stage_info = self.knowledge.get_stage_info(pregnancy_week)
            if stage_info:
                analysis['current_stage'] = stage_info
                analysis['trimester_info'] = self.knowledge.get_trimester_info(pregnancy_week)
        
        return analysis
    
    def generate_pregnancy_response(self, user_input: str, analysis: Dict[str, Any], user_id: str = "default") -> str:
        """Generate intelligent pregnancy-focused response based on analysis"""
        
        if not analysis['pregnancy_related'] and analysis['emotional_tone'] == 'neutral':
            return ""  # Let general AI handle non-pregnancy topics
        
        response_parts = []
        
        # Address emotional tone first
        if analysis['emotional_tone'] != 'neutral':
            emotional_support = self.knowledge.get_emotional_support(analysis['emotional_tone'])
            
            if analysis['emotional_tone'] == 'anxious':
                response_parts.append("I can hear the worry in your words, and that's completely understandable.")
            elif analysis['emotional_tone'] == 'excited':
                response_parts.append("Your excitement is wonderful! Pregnancy is such an amazing journey.")
            elif analysis['emotional_tone'] == 'overwhelmed':
                response_parts.append("Feeling overwhelmed is so normal during pregnancy. Let's take this one step at a time.")
            elif analysis['emotional_tone'] == 'sad':
                response_parts.append("I'm here with you through these difficult feelings. Pregnancy can bring up many emotions.")
            elif analysis['emotional_tone'] == 'confused':
                response_parts.append("It's okay to feel confused - there's so much information and so many changes happening.")
        
        # Handle medical attention needs
        if analysis['medical_attention']:
            response_parts.append("⚠️ **This sounds like something that needs medical attention. Please contact your healthcare provider or emergency services if this is urgent.**")
        
        # Address specific topics
        if 'symptoms' in analysis['topics']:
            response_parts.append("Let me help you understand what you're experiencing.")
        
        if 'nutrition' in analysis['topics']:
            nutrition_advice = self.knowledge.get_nutrition_advice(user_input)
            if nutrition_advice.get('specific_advice'):
                response_parts.append("Here's some nutrition guidance that might help:")
                for advice in nutrition_advice['specific_advice'][:3]:
                    response_parts.append(f"• {advice}")
        
        if 'exercise' in analysis['topics']:
            exercise_guidance = self.knowledge.get_exercise_guidance(concerns=user_input)
            if exercise_guidance.get('modifications'):
                response_parts.append("For exercise during pregnancy:")
                for mod in exercise_guidance['modifications'][:2]:
                    response_parts.append(f"• {mod}")
        
        # Add week-specific information if available
        if analysis.get('week_specific') and analysis.get('current_stage'):
            stage = analysis['current_stage']
            trimester_info = analysis.get('trimester_info', {})
            
            if len(response_parts) > 0:  # Only add if we have other content
                response_parts.append(f"\n**At {stage.week} weeks ({trimester_info.get('trimester', 'your')} trimester):**")
                
                # Add relevant stage information based on topics
                if any(topic in analysis['topics'] for topic in ['symptoms', 'medical']):
                    if stage.physical_changes:
                        response_parts.append(f"Common changes: {', '.join(stage.physical_changes[:2])}")
                
                if 'emotional' in analysis['topics'] or analysis['emotional_tone'] != 'neutral':
                    if stage.emotional_changes:
                        response_parts.append(f"Emotional patterns: {', '.join(stage.emotional_changes[:2])}")
                
                if 'baby_development' in analysis['topics']:
                    if stage.baby_development:
                        response_parts.append(f"Baby development: {', '.join(stage.baby_development[:2])}")
        
        # Add support strategies based on emotional support
        if analysis['emotional_tone'] != 'neutral':
            emotional_support = self.knowledge.get_emotional_support(analysis['emotional_tone'])
            if emotional_support.get('strategies'):
                response_parts.append("\n**Some strategies that might help:**")
                for strategy in emotional_support['strategies'][:3]:
                    response_parts.append(f"• {strategy}")
        
        # Combine all parts
        if response_parts:
            final_response = " ".join(response_parts)
            
            # Add supportive closing
            if analysis['emotional_tone'] in ['anxious', 'overwhelmed', 'sad']:
                final_response += "\n\n💙 Remember, you're not alone in this journey. I'm here to support you."
            elif analysis['emotional_tone'] == 'excited':
                final_response += "\n\n✨ It's beautiful to see your excitement about this journey!"
            else:
                final_response += "\n\n💙 I'm here if you have any other questions or concerns."
            
            return final_response
        
        return ""  # Return empty if no pregnancy-specific response needed
    
    def get_week_milestone_info(self, week: int) -> Optional[str]:
        """Get milestone information for specific pregnancy week"""
        stage = self.knowledge.get_stage_info(week)
        if not stage:
            return None
        
        trimester_info = self.knowledge.get_trimester_info(week)
        
        milestone_info = f"**Week {week} - {stage.stage_name}**\n"
        milestone_info += f"*{trimester_info['trimester']} Trimester*\n\n"
        
        if stage.baby_development:
            milestone_info += f"**Baby Development:** {', '.join(stage.baby_development)}\n\n"
        
        if stage.physical_changes:
            milestone_info += f"**What You Might Experience:** {', '.join(stage.physical_changes)}\n\n"
        
        if stage.medical_milestones:
            milestone_info += f"**Medical Milestones:** {', '.join(stage.medical_milestones)}\n\n"
        
        if stage.support_tips:
            milestone_info += f"**Support Tips:**\n"
            for tip in stage.support_tips[:3]:
                milestone_info += f"• {tip}\n"
        
        return milestone_info
    
    def analyze_symptoms_comprehensive(self, symptoms_text: str) -> Dict[str, Any]:
        """Comprehensive symptom analysis"""
        # Extract symptom keywords
        symptom_keywords = []
        for symptom in self.knowledge.symptoms_guide.keys():
            if symptom in symptoms_text.lower():
                symptom_keywords.append(symptom)
        
        # Use the knowledge base analysis
        analysis = self.knowledge.analyze_symptoms(symptom_keywords)
        
        # Add personalized response
        if analysis['medical_attention']:
            analysis['response'] = "⚠️ Some of what you're describing may need medical attention. Please contact your healthcare provider."
        elif analysis['normal_symptoms']:
            analysis['response'] = f"What you're experiencing with {', '.join(analysis['normal_symptoms'])} is common during pregnancy."
        else:
            analysis['response'] = "I'd be happy to help you understand what you're experiencing."
        
        return analysis
    
    def get_comprehensive_pregnancy_support(self, user_input: str, user_id: str = "default", pregnancy_week: int = 0) -> Optional[str]:
        """Main method to get comprehensive pregnancy intelligence support"""
        
        # Analyze the input
        analysis = self.analyze_pregnancy_context(user_input, user_id, pregnancy_week)
        
        # Generate response if pregnancy-related
        if analysis['pregnancy_related'] or analysis['emotional_tone'] != 'neutral':
            return self.generate_pregnancy_response(user_input, analysis, user_id)
        
        return None  # No pregnancy-specific response needed
    
    def update_user_profile(self, user_id: str, pregnancy_week: int = 0, preferences: Dict = None):
        """Update user profile with pregnancy information"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'pregnancy_week': 0,
                'preferences': {},
                'conversation_history': [],
                'concerns_tracking': [],
                'last_updated': datetime.now()
            }
        
        profile = self.user_profiles[user_id]
        
        if pregnancy_week > 0:
            profile['pregnancy_week'] = pregnancy_week
        
        if preferences:
            profile['preferences'].update(preferences)
        
        profile['last_updated'] = datetime.now()
    
    def get_user_pregnancy_summary(self, user_id: str) -> Optional[str]:
        """Get summary of user's pregnancy journey"""
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        week = profile.get('pregnancy_week', 0)
        
        if week == 0:
            return None
        
        stage = self.knowledge.get_stage_info(week)
        trimester_info = self.knowledge.get_trimester_info(week)
        
        summary = f"**Your Pregnancy Journey - Week {week}**\n"
        summary += f"*{trimester_info['trimester']} Trimester*\n\n"
        
        if stage:
            summary += f"**Current Stage:** {stage.stage_name}\n"
            summary += f"**Focus:** {trimester_info['focus']}\n\n"
            
            if stage.common_concerns:
                summary += f"**Common concerns at this stage:** {', '.join(stage.common_concerns[:2])}\n\n"
        
        summary += "💙 I'm here to support you through every step of this journey!"
        
        return summary


# ====== PREGNANCY INTELLIGENCE FACTORY ======

def create_pregnancy_intelligence() -> PregnancyIntelligence:
    """Factory function to create pregnancy intelligence instance"""
    return PregnancyIntelligence()


# ====== TESTING FUNCTION ======

def test_pregnancy_intelligence():
    """Test pregnancy intelligence functionality"""
    print("🧪 Testing Pregnancy Intelligence...")
    
    # Create instance
    pregnancy_ai = create_pregnancy_intelligence()
    
    # Test pregnancy context analysis
    test_inputs = [
        "I'm 20 weeks pregnant and feeling anxious about the anatomy scan",
        "I've been having terrible morning sickness and can't keep food down",
        "The baby isn't moving as much today and I'm worried",
        "I'm so excited to feel the baby kick!",
        "What should I be eating during my second trimester?"
    ]
    
    print("\n🔍 Testing Pregnancy Context Analysis:")
    for test_input in test_inputs:
        analysis = pregnancy_ai.analyze_pregnancy_context(test_input, "test_user", 20)
        print(f"  Input: '{test_input[:40]}...'")
        print(f"  Pregnancy-related: {analysis['pregnancy_related']}")
        print(f"  Emotional tone: {analysis['emotional_tone']}")
        print(f"  Topics: {analysis['topics']}")
        print(f"  Medical attention: {analysis['medical_attention']}")
        print()
    
    # Test response generation
    print("🤖 Testing Response Generation:")
    for test_input in test_inputs[:2]:
        analysis = pregnancy_ai.analyze_pregnancy_context(test_input, "test_user", 20)
        response = pregnancy_ai.generate_pregnancy_response(test_input, analysis, "test_user")
        if response:
            print(f"  Input: '{test_input[:30]}...'")
            print(f"  Response: '{response[:100]}...'")
            print()
    
    # Test week milestone info
    print("📅 Testing Week Milestone Info:")
    milestone = pregnancy_ai.get_week_milestone_info(20)
    if milestone:
        print(f"  Week 20 milestone: '{milestone[:100]}...'")
    
    # Test comprehensive support
    print("💝 Testing Comprehensive Support:")
    support_response = pregnancy_ai.get_comprehensive_pregnancy_support(
        "I'm 24 weeks and feeling overwhelmed about everything", 
        "test_user", 
        24
    )
    if support_response:
        print(f"  Support response: '{support_response[:100]}...'")
    
    print("\n✅ Pregnancy Intelligence testing complete!")


if __name__ == "__main__":
    # Run tests if executed directly
    test_pregnancy_intelligence()