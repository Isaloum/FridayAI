
# =====================================
# FILE 6: maternal_care/integration.py
# =====================================

from .SecureMaternalDatabase import SecureMaternalDatabase
from .MaternalHealthProfile import MaternalHealthProfile
from .OfflineCapableFriday import OfflineCapableFriday
from .PrivacyTrustManager import PrivacyTrustManager

def integrate_maternal_care_system(friday_ai):
    """
    Integrate the complete maternal care system with existing Friday
    """
    
    print("\n" + "="*50)
    print("🏥 FRIDAY MATERNAL CARE SYSTEM")
    print("Complete privacy-first maternal health platform")
    print("="*50)
    
    # Get user preferences
    print("\n🔒 Privacy & Security Setup:")
    use_password = input("Would you like to set a password for extra security? (y/n): ").lower() == 'y'
    
    user_password = None
    if use_password:
        user_password = input("Enter your secure password: ")
        print("✅ Password-based encryption enabled")
    
    offline_mode = input("Enable offline-only mode? (recommended for privacy) (y/n): ").lower() == 'y'
    
    if offline_mode:
        print("✅ Offline mode enabled - your data never leaves your device")
    else:
        print("ℹ️ Online mode - sync capabilities available")
    
    print("\n🏥 Initializing secure maternal health system...")
    
    try:
        # Initialize the system
        maternal_db = SecureMaternalDatabase(user_password, offline_mode)
        health_profile = MaternalHealthProfile(maternal_db)
        offline_friday = OfflineCapableFriday(friday_ai, maternal_db)
        privacy_manager = PrivacyTrustManager(maternal_db)
        
        # Add to Friday's capabilities
        friday_ai.maternal_care = {
            "database": maternal_db,
            "health_profile": health_profile,
            "offline_friday": offline_friday,
            "privacy_manager": privacy_manager,
            "user_id": None  # Will be set when user creates profile
        }
        
        print("\n✅ MATERNAL CARE SYSTEM READY!")
        print("✅ Your health data is encrypted and secure")
        print("✅ Friday can now provide personalized maternal support")
        print("✅ Complete privacy controls available")
        
        if offline_mode:
            print("✅ Fully offline capable - no internet required")
        
        # Show next steps
        print("\n📋 NEXT STEPS:")
        print("1. Create your maternal health profile: !create_profile")
        print("2. Track your pregnancy: !track_pregnancy")
        print("3. Monitor mental health: !track_mood")
        print("4. View privacy dashboard: !privacy")
        print("5. Export your data: !export_data")
        
        return friday_ai
        
    except Exception as e:
        print(f"❌ Failed to initialize maternal care system: {e}")
        return friday_ai

def create_maternal_profile_interactive(friday_ai):
    """Interactive profile creation"""
    
    if not hasattr(friday_ai, 'maternal_care'):
        print("❌ Maternal care system not initialized. Run integration first.")
        return
    
    print("\n👤 CREATING YOUR MATERNAL HEALTH PROFILE")
    print("This information is encrypted and stored securely on your device only.")
    print("-" * 50)
    
    # Collect profile information
    try:
        age = int(input("Your age: "))
        due_date = input("Expected due date (YYYY-MM-DD): ")
        conception_date = input("Conception date (YYYY-MM-DD, or press Enter to calculate): ")
        
        if not conception_date:
            from datetime import datetime, timedelta
            due_date_obj = datetime.strptime(due_date, "%Y-%m-%d")
            conception_date = (due_date_obj - timedelta(days=280)).strftime("%Y-%m-%d")
            print(f"Calculated conception date: {conception_date}")
        
        first_pregnancy = input("Is this your first pregnancy? (y/n): ").lower() == 'y'
        
        if not first_pregnancy:
            previous_pregnancies = int(input("Number of previous pregnancies: "))
        else:
            previous_pregnancies = 0
        
        # Medical history
        print("\n🏥 Medical History (press Enter to skip):")
        medical_conditions = input("Pre-existing conditions (comma-separated): ").split(',')
        medical_conditions = [condition.strip() for condition in medical_conditions if condition.strip()]
        
        medications = input("Current medications (comma-separated): ").split(',')
        medications = [med.strip() for med in medications if med.strip()]
        
        allergies = input("Allergies (comma-separated): ").split(',')
        allergies = [allergy.strip() for allergy in allergies if allergy.strip()]
        
        # Preferences
        print("\n⚙️ Preferences:")
        privacy_level = input("Privacy level (high/medium/low) [high]: ") or "high"
        communication_style = input("Communication style (warm/professional/casual) [warm]: ") or "warm"
        
        # Create profile
        profile_data = {
            "age": age,
            "due_date": due_date,
            "conception_date": conception_date,
            "first_pregnancy": first_pregnancy,
            "previous_pregnancies": previous_pregnancies,
            "medical_conditions": medical_conditions,
            "medications": medications,
            "allergies": allergies,
            "privacy_level": privacy_level,
            "communication_style": communication_style
        }
        
        user_id = friday_ai.maternal_care["health_profile"].create_user_profile(profile_data)
        friday_ai.maternal_care["user_id"] = user_id
        
        print(f"\n✅ Profile created successfully!")
        print(f"🆔 Your secure ID: {user_id[:8]}...")
        print("🔒 All data encrypted and stored locally")
        
        # Calculate current week
        current_week = friday_ai.maternal_care["health_profile"]._calculate_pregnancy_week(user_id)
        print(f"📅 You are currently at week {current_week} of your pregnancy")
        
        return user_id
        
    except Exception as e:
        print(f"❌ Error creating profile: {e}")
        return None

# =====================================
# Enhanced FridayAI Integration Commands
# Add these to your main FridayAI.py respond_to method
# =====================================

def add_maternal_care_commands_to_friday(friday_ai):
    """
    Add maternal care commands to Friday's command processing
    Add this to your main conversation loop in FridayAI.py
    """
    
    def process_maternal_command(user_input: str):
        """Process maternal care specific commands"""
        
        if not hasattr(friday_ai, 'maternal_care'):
            return "❌ Maternal care system not available. Please run integration first."
        
        # Profile management
        if user_input.lower().startswith("!create_profile"):
            return create_maternal_profile_interactive(friday_ai)
        
        # Privacy controls
        elif user_input.lower().startswith("!privacy"):
            if friday_ai.maternal_care["user_id"]:
                dashboard = friday_ai.maternal_care["privacy_manager"].get_privacy_dashboard(
                    friday_ai.maternal_care["user_id"]
                )
                return f"🔒 Privacy Dashboard:\n{json.dumps(dashboard, indent=2)}"
            return "❌ No profile found. Create profile first with !create_profile"
        
        # Data export
        elif user_input.lower().startswith("!export_data"):
            if friday_ai.maternal_care["user_id"]:
                data = friday_ai.maternal_care["privacy_manager"].export_user_data(
                    friday_ai.maternal_care["user_id"]
                )
                filename = f"friday_maternal_export_{datetime.now().strftime('%Y%m%d')}.json"
                with open(filename, 'w') as f:
                    f.write(data)
                return f"✅ Data exported to {filename}"
            return "❌ No profile found. Create profile first with !create_profile"
        
        # Quick pregnancy tracking
        elif user_input.lower().startswith("!track_pregnancy"):
            if friday_ai.maternal_care["user_id"]:
                return track_pregnancy_interactive(friday_ai)
            return "❌ No profile found. Create profile first with !create_profile"
        
        # Quick mood tracking
        elif user_input.lower().startswith("!track_mood"):
            if friday_ai.maternal_care["user_id"]:
                return track_mood_interactive(friday_ai)
            return "❌ No profile found. Create profile first with !create_profile"
        
        return None  # Not a maternal care command
    
    return process_maternal_command

def track_pregnancy_interactive(friday_ai):
    """Quick pregnancy tracking"""
    print("\n🤱 PREGNANCY TRACKING")
    print("-" * 30)
    
    try:
        # Quick symptom check
        symptoms_input = input("Current symptoms (comma-separated): ")
        symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
        
        energy = int(input("Energy level (1-10): ") or "5")
        sleep = int(input("Sleep quality (1-10): ") or "5")
        notes = input("Any notes: ")
        
        week_data = {
            "symptoms": symptoms,
            "energy_level": energy,
            "sleep_quality": sleep,
            "notes": notes
        }
        
        friday_ai.maternal_care["health_profile"].update_pregnancy_week(
            friday_ai.maternal_care["user_id"], week_data
        )
        
        return "✅ Pregnancy data tracked successfully!"
        
    except Exception as e:
        return f"❌ Error tracking pregnancy data: {e}"

def track_mood_interactive(friday_ai):
    """Quick mood tracking"""
    print("\n🧠 MENTAL HEALTH CHECK")
    print("-" * 30)
    
    try:
        anxiety = int(input("Anxiety level (0-10): ") or "0")
        stress = int(input("Stress level (0-10): ") or "0")
        wellbeing = int(input("Overall wellbeing (1-10): ") or "5")
        mood_desc = input("How are you feeling? ")
        
        mental_data = {
            "anxiety": anxiety,
            "stress": stress,
            "wellbeing": wellbeing,
            "mood_description": mood_desc
        }
        
        friday_ai.maternal_care["health_profile"].track_mental_health(
            friday_ai.maternal_care["user_id"], mental_data
        )
        
        return "✅ Mental health data tracked successfully!"
        
    except Exception as e:
        return f"❌ Error tracking mental health: {e}"

