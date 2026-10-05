import subprocess
import sys


def main():
    print("מתקין תלויות במידת הצורך...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("מפעיל את אפליקציית DocuBrain...")
    subprocess.run(["streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()