"""Douroub Lebanon: chat with the ReAct agent and its four personas.

    python react_app.py        # then open http://127.0.0.1:7860
"""
from douroub.ui import LAUNCH_OPTIONS, build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(**LAUNCH_OPTIONS)
