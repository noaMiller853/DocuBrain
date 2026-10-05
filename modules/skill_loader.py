import os
import yaml

def load_skill(skill_file_path: str):
    """טוען את קובץ ה-Skill מתוך נתיב YAML."""
    if os.path.exists(skill_file_path):
        with open(skill_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return None