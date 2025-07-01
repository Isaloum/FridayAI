# DomainAdapterCore.py

from pregnancy_support.core.PregnancySupportCore import PregnancySupportCore
from IntentionReflectionCore import IntentionReflectionCore

class DomainAdapterCore:
    def __init__(self, memory, emotion_core, intent_engine):
        self.memory = memory
        self.emotion_core = emotion_core
        self.intent_engine = intent_engine

        self.active_domain = None
        self.ability_modules = {}
        self.behavior_mode = "default"
        self.context_data = {}

        self.reflection_core = IntentionReflectionCore(
            memory_core=self.memory,
            emotion_core=self.emotion_core,
            goal_log=self.intent_engine.get_goal_log()
        )


    def load_domain(self, domain_name, domain_context):
        self.active_domain = domain_name
        self.context_data[domain_name] = domain_context

        if domain_name == "pregnancy":
            self.pregnancy_module = PregnancySupportCore()
            self.attach_ability_modules("pregnancy", [
                self.pregnancy_module.respond_to_feeling
            ])

        print(f"[DomainAdapter] Loaded domain: {domain_name}")


    def attach_ability_modules(self, domain_name, modules):
        if domain_name not in self.ability_modules:
            self.ability_modules[domain_name] = []
        self.ability_modules[domain_name].extend(modules)
        #print(f"[DomainAdapter] Attached {len(modules)} modules to domain: {domain_name}")

    def set_behavioral_modulation(self, mode):
        self.behavior_mode = mode
        print(f"[DomainAdapter] Behavior modulation set to: {mode}")

    def domain_context_reflection(self):
        return {
            "active_domain": self.active_domain,
            "behavior_mode": self.behavior_mode,
            "context": self.context_data.get(self.active_domain, {}),
            "abilities": [m.__name__ for m in self.ability_modules.get(self.active_domain, [])]
        }

    def execute_ability(self, ability_name, *args, **kwargs):
        if not self.active_domain:
            raise RuntimeError("No active domain loaded.")
        for mod in self.ability_modules.get(self.active_domain, []):
            if mod.__name__ == ability_name:
                return mod(*args, **kwargs)
        raise ValueError(f"Ability '{ability_name}' not found in active domain '{self.active_domain}'.")

    def flush_and_reset(self):
        print(f"[DomainAdapter] Flushing domain: {self.active_domain}")
        self.active_domain = None
        self.behavior_mode = "default"
        self.context_data.clear()
        self.ability_modules.clear()
        print("[DomainAdapter] Reset complete.")

    def run_intention_reflection(self):
        prompts = self.reflection_core.prompt_realignment()
        score = self.reflection_core.identity_cohesion_score()
        return prompts, score
