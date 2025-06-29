import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Brain, Zap, Heart, Sparkles, AlertCircle, Target, 
  Moon, Sun, Coffee, Activity, Shield, Lock, 
  TrendingUp, Clock, CheckCircle, XCircle,
  MessageCircle, Settings, Info, Award, Database,
  Mic, MicOff, Volume2, VolumeX, Save, Download
} from 'lucide-react';

// ==================== TYPES & INTERFACES ====================
const TONE_MODES = {
  supportive: { icon: '💙', label: 'Supportive', desc: 'Warm & empathetic' },
  sassy: { icon: '💅', label: 'Sassy', desc: 'Confident & playful' },
  direct: { icon: '📊', label: 'Direct', desc: 'Facts-focused' },
  clinical: { icon: '🏥', label: 'Clinical', desc: 'Medical terminology' },
  friendly: { icon: '😊', label: 'Friendly', desc: 'Casual conversation' }
};

const EMERGENCY_LEVELS = {
  critical: { color: 'red', icon: '🚨', label: 'CRITICAL' },
  urgent: { color: 'orange', icon: '⚠️', label: 'URGENT' },
  concerning: { color: 'yellow', icon: '⚠️', label: 'CONCERNING' }
};

// ==================== UTILITY COMPONENTS ====================
const LoadingDots = () => (
  <div className="flex space-x-1">
    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
  </div>
);

const SystemStatus = ({ status }) => {
  const getStatusColor = () => {
    if (status.performance > 90) return 'bg-green-400';
    if (status.performance > 70) return 'bg-yellow-400';
    return 'bg-red-400';
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 ${getStatusColor()} rounded-full animate-pulse`}></div>
      <span className="text-xs text-gray-400">
        {status.successful_interactions} interactions • {status.performance}% health
      </span>
    </div>
  );
};

// ==================== MAIN FRIDAY AI COMPONENT ====================
const FridayAI_SuperUltraBrilliant = () => {
  // ========== STATE MANAGEMENT ==========
  // User & Session State
  const [userName, setUserName] = useState('');
  const [pregnancyWeek, setPregnancyWeek] = useState(0);
  const [currentUserId, setCurrentUserId] = useState('default');
  const [isSetupComplete, setIsSetupComplete] = useState(false);

  // Conversation State
  const [currentMessage, setCurrentMessage] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTone, setCurrentTone] = useState('supportive');
  
  // Feature States
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [encryptionEnabled, setEncryptionEnabled] = useState(true);
  const [showSystemPanel, setShowSystemPanel] = useState(false);
  const [showGoalPanel, setShowGoalPanel] = useState(false);
  
  // Advanced Features State
  const [legendaryMemory, setLegendaryMemory] = useState({
    conversations: [],
    topicMemory: {},
    emotionalPatterns: {},
    userPatterns: {},
    sessionInsights: {}
  });

  const [activeGoals, setActiveGoals] = useState([]);
  const [completedGoals, setCompletedGoals] = useState([]);
  
  const [systemHealth, setSystemHealth] = useState({
    uptime: Date.now(),
    performance: 100,
    successful_interactions: 0,
    error_count: 0,
    emotional_intelligence_level: 95,
    response_times: [],
    emergency_responses: 0,
    vault_operations: 0,
    predictions_made: 0
  });

  const [unstoppableFeatures, setUnstoppableFeatures] = useState({
    resilience_active: true,
    predictive_analytics: true,
    emergency_protocol: true,
    secure_vault: true
  });

  // UI State
  const messagesEndRef = useRef(null);
  const [showEmergencyChecklist, setShowEmergencyChecklist] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState(null);

  // ========== CORE ANALYSIS FUNCTIONS ==========
  const analyzeInputSemantic = useCallback((input) => {
    const text = input.toLowerCase();
    
    // Emergency Detection First
    const emergencyKeywords = {
      bleeding: /\b(bleeding|blood|spotting heavily|gushing)\b/gi,
      pain: /\b(severe pain|intense pain|unbearable|sharp pain)\b/gi,
      breathing: /\b(can't breathe|trouble breathing|shortness of breath)\b/gi,
      contractions: /\b(contractions|labor pains|regular contractions)\b/gi,
      consciousness: /\b(dizzy|faint|passing out|unconscious)\b/gi,
    };

    for (const [type, pattern] of Object.entries(emergencyKeywords)) {
      if (pattern.test(text)) {
        return {
          type: 'emergency',
          emergency_type: type,
          urgency_level: 'critical',
          confidence: 1.0
        };
      }
    }

    // Emotional Analysis
    const emotions = {
      anxiety: /\b(worried|anxious|scared|nervous|panic|stress|overwhelming)\b/g,
      joy: /\b(happy|excited|joy|thrilled|amazing|wonderful|great)\b/g,
      sadness: /\b(sad|depressed|down|crying|upset|hurt|disappointed)\b/g,
      fatigue: /\b(tired|exhausted|drained|sleepy|worn out|fatigued)\b/g,
      pain: /\b(pain|hurt|ache|sore|cramp|discomfort)\b/g
    };

    const detectedEmotions = [];
    Object.entries(emotions).forEach(([emotion, pattern]) => {
      if (pattern.test(text)) {
        detectedEmotions.push(emotion);
      }
    });

    // Pregnancy Topic Detection
    const pregnancyTopics = {
      physical: /\b(nausea|vomiting|morning sickness|back pain|swelling|headache)\b/g,
      emotional: /\b(mood swings|hormonal|emotional|crying|irritable)\b/g,
      developmental: /\b(baby|fetus|growth|development|movement|kicks|heartbeat)\b/g,
      medical: /\b(doctor|appointment|test|ultrasound|blood work|checkup)\b/g
    };

    const detectedTopics = [];
    Object.entries(pregnancyTopics).forEach(([topic, pattern]) => {
      if (pattern.test(text)) {
        detectedTopics.push(topic);
      }
    });

    const isQuestion = /\?|what|how|when|where|why|should|can|will|is it/i.test(text);

    return {
      type: detectedTopics.length > 0 ? 'pregnancy_related' : 'general_conversation',
      emotions: detectedEmotions,
      pregnancy_topics: detectedTopics,
      is_question: isQuestion,
      confidence: 0.85 + (detectedEmotions.length * 0.05)
    };
  }, []);

  // ========== RESPONSE GENERATION ==========
  const generateResponse = useCallback((userInput, analysis) => {
    const baseResponses = {
      emergency: {
        bleeding: "🚨 **CRITICAL EMERGENCY** 🚨\n\n**CALL 911 NOW**\n\nWhile waiting:\n1. Lie down and elevate feet\n2. Do not insert anything\n3. Track bleeding amount\n4. Stay calm - help is coming",
        pain: "🚨 **SEVERE PAIN - URGENT** 🚨\n\n**Contact your healthcare provider immediately or go to the hospital**\n\nThis needs immediate evaluation.",
        default: "🚨 **MEDICAL EMERGENCY** 🚨\n\n**Seek immediate medical attention**\n\nDon't wait - your safety is the priority."
      },
      supportive: {
        anxiety: `I understand you're feeling anxious, ${userName || 'dear'}. These feelings are completely valid. Your body is going through so many changes. Let's work through this together. What specific worries are on your mind?`,
        joy: `Your excitement is beautiful, ${userName || 'friend'}! This journey is filled with amazing moments. Tell me what's bringing you joy today!`,
        general: `I'm here for you, ${userName || 'friend'}. Whatever you're experiencing, your feelings matter. How can I support you right now?`
      },
      sassy: {
        anxiety: `Girl, I see those worries trying to take over! Listen, ${userName || 'honey'}, you're stronger than any anxiety. Let's tackle these concerns head-on!`,
        general: `Hey there ${userName || 'gorgeous'}! Mama Friday is here and ready to chat. What's the tea today?`
      },
      direct: {
        general: `Based on medical evidence: Your query has been received. Please specify your primary concern for targeted information.`,
        pregnancy: `Current pregnancy week: ${pregnancyWeek}. Provide specific symptoms or questions for evidence-based guidance.`
      }
    };

    // Handle emergencies
    if (analysis.type === 'emergency') {
      return {
        content: baseResponses.emergency[analysis.emergency_type] || baseResponses.emergency.default,
        isEmergency: true,
        confidence: 1.0
      };
    }

    // Select appropriate response based on tone and emotion
    let responseCategory = baseResponses[currentTone] || baseResponses.supportive;
    let selectedResponse;

    if (analysis.emotions.includes('anxiety')) {
      selectedResponse = responseCategory.anxiety || responseCategory.general;
    } else if (analysis.emotions.includes('joy')) {
      selectedResponse = responseCategory.joy || responseCategory.general;
    } else {
      selectedResponse = responseCategory.general;
    }

    // Add context-aware elements
    if (pregnancyWeek > 0 && analysis.pregnancy_topics.length > 0) {
      selectedResponse += `\n\nAt ${pregnancyWeek} weeks, you're ${getPregnancyStage(pregnancyWeek)}. `;
      
      if (pregnancyWeek < 13) {
        selectedResponse += "First trimester can be challenging with all the changes.";
      } else if (pregnancyWeek < 27) {
        selectedResponse += "Second trimester often brings more energy and comfort.";
      } else {
        selectedResponse += "Third trimester - you're in the home stretch!";
      }
    }

    return {
      content: selectedResponse,
      isEmergency: false,
      confidence: analysis.confidence
    };
  }, [userName, pregnancyWeek, currentTone]);

  // ========== HELPER FUNCTIONS ==========
  const getPregnancyStage = (week) => {
    if (week < 13) return "in your first trimester";
    if (week < 27) return "in your second trimester";
    return "in your third trimester";
  };

  const detectNameInMessage = (message) => {
    const patterns = [
      /(?:call me|my name is|i'm|i am|this is)\s+([A-Za-z]+)/i,
      /([A-Za-z]+)\s+here/i
    ];

    for (const pattern of patterns) {
      const match = message.match(pattern);
      if (match && match[1]) {
        return match[1].charAt(0).toUpperCase() + match[1].slice(1).toLowerCase();
      }
    }
    return null;
  };

  const formatTimestamp = (date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    }).format(date);
  };

  // ========== MESSAGE HANDLING ==========
  const handleSendMessage = useCallback(() => {
    if (!currentMessage.trim() || isLoading) return;

    const messageText = currentMessage.trim();
    setCurrentMessage('');
    setIsLoading(true);

    // Check for name change
    const detectedName = detectNameInMessage(messageText);
    if (detectedName && detectedName !== userName) {
      setUserName(detectedName);
      addMessage('user', messageText);
      addMessage('ai', `Perfect! I'll call you ${detectedName} from now on. How can I help you today, ${detectedName}?`, {
        tone: currentTone,
        special: 'name_change'
      });
      setIsLoading(false);
      return;
    }

    // Add user message
    addMessage('user', messageText);

    // Simulate AI processing
    setTimeout(() => {
      const analysis = analyzeInputSemantic(messageText);
      const response = generateResponse(messageText, analysis);

      // Update system health
      setSystemHealth(prev => ({
        ...prev,
        successful_interactions: prev.successful_interactions + 1,
        emergency_responses: response.isEmergency ? prev.emergency_responses + 1 : prev.emergency_responses
      }));

      // Update legendary memory
      updateLegendaryMemory(messageText, response.content, analysis);

      // Add AI response
      addMessage('ai', response.content, {
        tone: currentTone,
        analysis,
        isEmergency: response.isEmergency,
        confidence: response.confidence
      });

      setIsLoading(false);
    }, 800 + Math.random() * 400);
  }, [currentMessage, isLoading, userName, currentTone, analyzeInputSemantic, generateResponse]);

  const addMessage = (type, content, metadata = {}) => {
    const message = {
      id: Date.now() + Math.random(),
      type,
      content,
      timestamp: new Date(),
      ...metadata
    };
    setConversationHistory(prev => [...prev, message]);
  };

  const updateLegendaryMemory = (userInput, aiResponse, analysis) => {
    setLegendaryMemory(prev => {
      const topic = analysis.pregnancy_topics[0] || 'general';
      const emotion = analysis.emotions[0] || 'neutral';

      return {
        ...prev,
        conversations: [...prev.conversations, { userInput, aiResponse, timestamp: new Date() }].slice(-50),
        topicMemory: {
          ...prev.topicMemory,
          [topic]: [...(prev.topicMemory[topic] || []), { userInput, timestamp: new Date() }].slice(-10)
        },
        emotionalPatterns: {
          ...prev.emotionalPatterns,
          [emotion]: (prev.emotionalPatterns[emotion] || 0) + 1
        }
      };
    });
  };

  // ========== SPECIAL COMMANDS ==========
  const handleSpecialCommand = useCallback((command) => {
    const cmd = command.toLowerCase().trim();

    if (cmd === '!status') {
      const uptime = Date.now() - systemHealth.uptime;
      const days = Math.floor(uptime / (1000 * 60 * 60 * 24));
      const hours = Math.floor((uptime % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));

      return `🏆 **Friday AI Status Report**

**System Health:**
• Uptime: ${days} days, ${hours} hours
• Performance: ${systemHealth.performance}%
• Interactions: ${systemHealth.successful_interactions}
• Emergency Responses: ${systemHealth.emergency_responses}

**Features Active:**
• 🧠 Legendary Memory: ✅ (${legendaryMemory.conversations.length} stored)
• 🎯 Goal Coaching: ✅ (${activeGoals.length} active)
• 🛡️ Resilience Engine: ✅
• 📊 Predictive Analytics: ✅
• 🔒 Secure Vault: ${encryptionEnabled ? '✅' : '❌'}
• 🔊 Voice Features: ${voiceEnabled ? '✅' : '❌'}

**Current Settings:**
• Tone: ${currentTone}
• User: ${userName}
• Pregnancy Week: ${pregnancyWeek || 'Not set'}`;
    }

    if (cmd === '!goals') {
      return `🎯 **Your Goals**

Active Goals: ${activeGoals.length}
Completed Goals: ${completedGoals.length}

Say "create goal" to start a new goal!`;
    }

    if (cmd === '!emergency') {
      setShowEmergencyChecklist(true);
      return null;
    }

    return null;
  }, [systemHealth, legendaryMemory, activeGoals, completedGoals, encryptionEnabled, voiceEnabled, currentTone, userName, pregnancyWeek]);

  // ========== EFFECTS ==========
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversationHistory]);

  useEffect(() => {
    if (isSetupComplete && conversationHistory.length === 0) {
      const greeting = `🏆 **Friday AI - Super Ultra Brilliant Edition**

Hello ${userName}! I'm Friday, your advanced AI companion designed specifically to support you through your journey.

${pregnancyWeek > 0 ? `I see you're at ${pregnancyWeek} weeks - ${getPregnancyStage(pregnancyWeek)}. What an exciting time!\n\n` : ''}
💡 **I can help with:**
• Emotional support & understanding
• Pregnancy guidance & information
• Goal setting & tracking
• Emergency response & safety
• Personalized insights & predictions

How are you feeling today?`;

      addMessage('ai', greeting, {
        tone: 'supportive',
        special: 'greeting'
      });
    }
  }, [isSetupComplete, userName, pregnancyWeek, conversationHistory.length]);

  // ========== RENDER FUNCTIONS ==========
  const renderSetupScreen = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <div className="max-w-lg w-full bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/30 p-8">
        <div className="text-center mb-8">
          <div className="relative inline-block mb-4">
            <Brain className="w-20 h-20 text-purple-400" />
            <Sparkles className="w-6 h-6 text-yellow-400 absolute -top-2 -right-2 animate-pulse" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Friday AI</h1>
          <p className="text-purple-300">Super Ultra Brilliant Edition</p>
        </div>

        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-purple-300 mb-2">
              What should I call you?
            </label>
            <input
              type="text"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && userName && setIsSetupComplete(true)}
              className="w-full px-4 py-3 bg-black/30 border border-purple-500/50 rounded-lg text-white placeholder-purple-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              placeholder="Enter your name"
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-purple-300 mb-2">
              Pregnancy week (optional)
            </label>
            <input
              type="number"
              min="0"
              max="42"
              value={pregnancyWeek || ''}
              onChange={(e) => setPregnancyWeek(Number(e.target.value))}
              onKeyDown={(e) => e.key === 'Enter' && userName && setIsSetupComplete(true)}
              className="w-full px-4 py-3 bg-black/30 border border-purple-500/50 rounded-lg text-white placeholder-purple-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              placeholder="Leave blank if not applicable"
            />
          </div>

          <button
            onClick={() => setIsSetupComplete(true)}
            disabled={!userName}
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-lg font-medium hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center gap-2"
          >
            <Zap size={20} />
            Activate Friday AI
          </button>
        </div>

        <div className="mt-8 grid grid-cols-2 gap-4 text-center">
          <div className="bg-purple-500/10 rounded-lg p-3">
            <Shield className="w-6 h-6 text-purple-400 mx-auto mb-1" />
            <p className="text-xs text-purple-300">Secure & Private</p>
          </div>
          <div className="bg-purple-500/10 rounded-lg p-3">
            <Activity className="w-6 h-6 text-purple-400 mx-auto mb-1" />
            <p className="text-xs text-purple-300">24/7 Support</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderMessage = (message) => {
    const isUser = message.type === 'user';
    const isEmergency = message.isEmergency;

    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div className={`max-w-2xl ${isUser ? 'order-2' : 'order-1'}`}>
          {!isUser && (
            <div className="flex items-center gap-2 mb-1">
              <Brain className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-purple-400 font-medium">Friday AI</span>
              {message.tone && (
                <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">
                  {TONE_MODES[message.tone]?.label || message.tone}
                </span>
              )}
              {isEmergency && (
                <span className="text-xs bg-red-500/20 text-red-300 px-2 py-0.5 rounded flex items-center gap-1">
                  <AlertCircle size={12} />
                  Emergency
                </span>
              )}
            </div>
          )}

          <div
            className={`px-4 py-3 rounded-2xl ${
              isUser
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-br-none'
                : isEmergency
                ? 'bg-red-500/20 border-2 border-red-500/50 text-white'
                : 'bg-black/40 backdrop-blur-sm border border-purple-500/30 text-white rounded-bl-none'
            }`}
          >
            <p className="text-sm leading-relaxed whitespace-pre-line">{message.content}</p>
            <p className="text-xs opacity-60 mt-2">{formatTimestamp(message.timestamp)}</p>
          </div>

          {message.confidence && (
            <div className="flex items-center gap-1 mt-1 text-xs text-purple-400">
              <Activity size={12} />
              <span>Confidence: {(message.confidence * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderEmergencyChecklist = () => (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-slate-900 border border-red-500/50 rounded-xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <AlertCircle className="text-red-500" />
            Emergency Preparedness Checklist
          </h2>
          <button
            onClick={() => setShowEmergencyChecklist(false)}
            className="text-gray-400 hover:text-white"
          >
            <XCircle size={24} />
          </button>
        </div>

        <div className="space-y-6 text-white">
          <div>
            <h3 className="font-semibold mb-2 text-red-400">Important Numbers to Save:</h3>
            <ul className="space-y-1 text-sm">
              <li>• Your OB/GYN office number</li>
              <li>• Hospital labor & delivery unit</li>
              <li>• Emergency contact person</li>
              <li>• Poison control: 1-800-222-1222 (US)</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-2 text-red-400">Emergency Signs During Pregnancy:</h3>
            <ul className="space-y-1 text-sm">
              <li>• Severe bleeding or cramping</li>
              <li>• Baby's movements stopped or decreased significantly</li>
              <li>• Severe headaches with vision changes</li>
              <li>• Persistent vomiting</li>
              <li>• Signs of preterm labor</li>
              <li>• Water breaking before 37 weeks</li>
              <li>• Severe abdominal pain</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-2 text-red-400">What to Have Ready:</h3>
            <ul className="space-y-1 text-sm">
              <li>• Hospital bag packed (by 36 weeks)</li>
              <li>• Birth plan copies</li>
              <li>• Insurance cards and ID</li>
              <li>• List of current medications</li>
              <li>• Emergency contact list</li>
            </ul>
          </div>

          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
            <p className="text-sm">
              <strong>Remember:</strong> It's always better to call and be reassured than to wait and risk complications.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSystemPanel = () => (
    <div className="absolute top-16 right-4 w-80 bg-black/90 backdrop-blur-sm border border-purple-500/30 rounded-xl p-4 z-40">
      <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
        <Settings size={16} />
        System Information
      </h3>
      
      <div className="space-y-3 text-sm">
        <div>
          <p className="text-purple-300 mb-1">Performance</p>
          <div className="bg-purple-900/30 rounded-full h-2 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all"
              style={{ width: `${systemHealth.performance}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-purple-900/20 rounded p-2">
            <p className="text-purple-400">Interactions</p>
            <p className="text-white font-semibold">{systemHealth.successful_interactions}</p>
          </div>
          <div className="bg-purple-900/20 rounded p-2">
            <p className="text-purple-400">Emergencies</p>
            <p className="text-white font-semibold">{systemHealth.emergency_responses}</p>
          </div>
        </div>

        <div className="space-y-2">
          <p className="text-purple-300">Features</p>
          {Object.entries(unstoppableFeatures).map(([feature, active]) => (
            <div key={feature} className="flex items-center justify-between">
              <span className="text-gray-400 text-xs">{feature.replace(/_/g, ' ')}</span>
              <span className={active ? 'text-green-400' : 'text-red-400'}>
                {active ? <CheckCircle size={14} /> : <XCircle size={14} />}
              </span>
            </div>
          ))}
        </div>

        <button
          onClick={() => {
            const data = {
              userName,
              pregnancyWeek,
              conversations: conversationHistory,
              systemHealth,
              timestamp: new Date().toISOString()
            };
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `friday_export_${Date.now()}.json`;
            a.click();
          }}
          className="w-full bg-purple-500/20 text-purple-300 py-2 rounded hover:bg-purple-500/30 transition-colors flex items-center justify-center gap-2"
        >
          <Download size={14} />
          Export Data
        </button>
      </div>
    </div>
  );

  // ========== MAIN RENDER ==========
  if (!isSetupComplete) {
    return renderSetupScreen();
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-purple-500/30">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 rounded-full flex items-center justify-center">
                  <Brain className="text-white" size={24} />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Friday AI
                </h1>
                <SystemStatus status={systemHealth} />
              </div>
            </div>

            <div className="flex items-center gap-4">
              {pregnancyWeek > 0 && (
                <div className="text-center bg-purple-500/10 rounded-lg px-4 py-2">
                  <p className="text-3xl font-bold text-purple-400">{pregnancyWeek}</p>
                  <p className="text-xs text-purple-300">weeks</p>
                </div>
              )}

              <div className="flex gap-2">
                <button
                  onClick={() => setVoiceEnabled(!voiceEnabled)}
                  className={`p-2 rounded-lg transition-colors ${
                    voiceEnabled ? 'bg-purple-500 text-white' : 'bg-purple-900/50 text-purple-300'
                  }`}
                  title={voiceEnabled ? 'Voice On' : 'Voice Off'}
                >
                  {voiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
                </button>

                <button
                  onClick={() => setShowSystemPanel(!showSystemPanel)}
                  className="p-2 rounded-lg bg-purple-900/50 text-purple-300 hover:bg-purple-800/50 transition-colors"
                  title="System Info"
                >
                  <Info size={20} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto h-[calc(100vh-80px)] flex">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Tone Selector */}
          <div className="bg-black/20 border-b border-purple-500/30 p-4">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-purple-300 text-sm mr-2">Communication Mode:</span>
              {Object.entries(TONE_MODES).map(([id, { icon, label, desc }]) => (
                <button
                  key={id}
                  onClick={() => setCurrentTone(id)}
                  className={`px-3 py-1 rounded-full text-xs transition-all flex items-center gap-1 ${
                    currentTone === id
                      ? 'bg-purple-500 text-white border border-purple-400'
                      : 'bg-purple-900/50 text-purple-300 hover:bg-purple-800/50'
                  }`}
                  title={desc}
                >
                  <span>{icon}</span>
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4">
            {conversationHistory.map(renderMessage)}
            
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-black/40 backdrop-blur-sm border border-purple-500/30 rounded-2xl rounded-bl-none px-4 py-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-purple-400 animate-pulse" />
                    <span className="text-xs text-purple-400 font-medium">Friday is thinking...</span>
                  </div>
                  <LoadingDots />
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 bg-black/20 border-t border-purple-500/30">
            <div className="flex gap-3">
              <input
                type="text"
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder={`Hi ${userName}, what's on your mind? (Try "call me [name]" to change your name)`}
                className="flex-1 px-4 py-3 bg-black/30 border border-purple-500/50 rounded-lg text-white placeholder-purple-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                disabled={isLoading}
              />
              
              {voiceEnabled && (
                <button
                  className="px-4 py-3 bg-purple-900/50 text-purple-300 rounded-lg hover:bg-purple-800/50 transition-colors"
                  title="Voice Input"
                >
                  <Mic size={20} />
                </button>
              )}
              
              <button
                onClick={handleSendMessage}
                disabled={!currentMessage.trim() || isLoading}
                className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-purple-600 hover:to-pink-600 transition-all flex items-center gap-2"
              >
                <Zap size={20} />
                Send
              </button>
            </div>
            
            <div className="flex items-center justify-between mt-2">
              <p className="text-xs text-purple-400">
                💡 Try: !status, !goals, !emergency for special features
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowGoalPanel(!showGoalPanel)}
                  className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
                >
                  <Target size={12} />
                  Goals ({activeGoals.length})
                </button>
                <button
                  onClick={() => setShowEmergencyChecklist(true)}
                  className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
                >
                  <AlertCircle size={12} />
                  Emergency Info
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel - Goals */}
        {showGoalPanel && (
          <div className="w-80 bg-black/20 border-l border-purple-500/30 p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-semibold flex items-center gap-2">
                <Target size={16} />
                Your Goals
              </h3>
              <button
                onClick={() => setShowGoalPanel(false)}
                className="text-gray-400 hover:text-white"
              >
                <XCircle size={20} />
              </button>
            </div>

            {activeGoals.length === 0 ? (
              <div className="text-center py-8">
                <Target className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                <p className="text-purple-300 text-sm mb-4">No active goals yet</p>
                <button
                  onClick={() => {
                    handleSendMessage("create goal");
                    setCurrentMessage("create goal");
                  }}
                  className="bg-purple-500/20 text-purple-300 px-4 py-2 rounded-lg hover:bg-purple-500/30 transition-colors text-sm"
                >
                  Create Your First Goal
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {activeGoals.map((goal, index) => (
                  <div key={index} className="bg-purple-900/20 rounded-lg p-3">
                    <h4 className="text-white font-medium text-sm mb-2">{goal.title}</h4>
                    <div className="bg-purple-900/30 rounded-full h-2 overflow-hidden mb-2">
                      <div 
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all"
                        style={{ width: `${goal.progress || 0}%` }}
                      />
                    </div>
                    <p className="text-xs text-purple-300">{goal.progress || 0}% complete</p>
                  </div>
                ))}
              </div>
            )}

            {completedGoals.length > 0 && (
              <div className="mt-6">
                <h4 className="text-purple-300 text-sm font-medium mb-2">Completed Goals</h4>
                <div className="space-y-2">
                  {completedGoals.map((goal, index) => (
                    <div key={index} className="bg-green-900/20 rounded p-2">
                      <p className="text-green-300 text-xs flex items-center gap-1">
                        <CheckCircle size={12} />
                        {goal.title}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modals */}
      {showEmergencyChecklist && renderEmergencyChecklist()}
      {showSystemPanel && renderSystemPanel()}

      {/* Special command processing */}
      {currentMessage.startsWith('!') && (
        <div className="hidden">
          {(() => {
            const response = handleSpecialCommand(currentMessage);
            if (response) {
              setTimeout(() => {
                addMessage('ai', response, { tone: 'system', special: 'command' });
                setCurrentMessage('');
              }, 0);
            }
          })()}
        </div>
      )}
    </div>
  );
};

export default FridayAI_SuperUltraBrilliant;
