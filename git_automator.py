#!/usr/bin/env python3
import subprocess
import os
from datetime import datetime
from openai import OpenAI
import sys

class GitAutomator:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=self.openai_api_key)

    def get_git_changes(self):
        """Récupère les modifications Git en attente"""
        try:
            result = subprocess.run(['git', 'diff', '--staged', '--name-status'], 
                                 capture_output=True, text=True)
            staged = result.stdout.strip()
            
            result = subprocess.run(['git', 'diff', '--name-status'], 
                                 capture_output=True, text=True)
            unstaged = result.stdout.strip()
            
            return staged + '\n' + unstaged
        except Exception as e:
            print(f"Erreur lors de la récupération des modifications Git: {e}")
            return None

    def generate_commit_message(self, changes):
        """Génère un message de commit avec GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Tu es un assistant qui génère des messages de commit concis et descriptifs en anglais. Format: scope: action détaillée (par exemple 'florence_vision_client: add prompt parameter in function analyze_image'). Évite les préfixes comme 'chore', 'feat', etc."},
                    {"role": "user", "content": f"Génère un message de commit pour ces modifications:\n{changes}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Erreur lors de la génération du message: {e}")
            return datetime.now().strftime("update: %Y-%m-%d %H:%M")

    def execute_git_commands(self):
        """Exécute la séquence de commandes Git"""
        try:
            # git add .
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Récupère les modifications
            changes = self.get_git_changes()
            if not changes:
                print("Aucune modification à commiter")
                return
            
            # Génère le message de commit
            commit_message = self.generate_commit_message(changes)
            
            # git commit
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # git push
            subprocess.run(['git', 'push'], check=True)
            
            print(f"✅ Push réussi avec le message: {commit_message}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de l'exécution des commandes Git: {e}")
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    # Récupère la clé API depuis une variable d'environnement
    api_key = os.getenv('OPENAI_API_GIT_AUTOMATOR_KEY')
    if not api_key:
        print("❌ OPENAI_API_GIT_AUTOMATOR_KEY non définie dans les variables d'environnement")
        sys.exit(1)
    
    automator = GitAutomator(api_key)
    automator.execute_git_commands()