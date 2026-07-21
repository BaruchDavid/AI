# End-to-End Agentic AI Chatbot

Ein End-to-End Agentic-AI-Chatbot-Projekt auf Basis von **LangChain** und **LangGraph**, mit einem **Streamlit**-Frontend. Der Bot verarbeitet Nutzeranfragen nicht linear über einen einzelnen LLM-Call, sondern über einen **Graph-Workflow** aus mehreren Knoten (Nodes), die jeweils eigene Logik und/oder LLM-Aufrufe kapseln.

## Architektur

Das Projekt gliedert sich in drei unabhängige Komponentenbereiche:

**Frontend** – Eine Streamlit-Oberfläche mit Chat-Eingabefeld, die Nutzereingaben entgegennimmt und an den Workflow übergibt sowie die Antworten des Agenten anzeigt.

**Workflow (Graph)** – Das Herzstück des Agenten, umgesetzt mit LangGraph. Der Workflow besteht aus:

- **Nodes**: einzelne Verarbeitungsschritte im Graphen, die jeweils eine unabhängige Funktion (teilweise mit eigenem LLM-Aufruf) ausführen.
- **Edges**: Übergänge zwischen den Nodes, die den Kontrollfluss festlegen – inklusive bedingter Verzweigungen (z. B. der Split in zwei parallele Pfade in der Mitte des Graphen), die je nach Zwischenergebnis unterschiedliche Nodes ansteuern.
- **State**: der gemeinsame Zustand, der zwischen den Nodes weitergereicht wird und alle relevanten Informationen (Nutzereingabe, Zwischenergebnisse, LLM-Antworten etc.) über den gesamten Ablauf hinweg trägt.

**Independent Components** – Die eigentliche Fachlogik ist bewusst von der Graph-Struktur entkoppelt und in eigenständigen Funktionen ausgelagert, die jeweils mit einem oder mehreren LLMs interagieren (`function + LLM's`, `function2 + LLMs`, `function3 + LLM`). Diese Funktionen werden von den Nodes aufgerufen, sind aber unabhängig testbar und wiederverwendbar.

```
User (Streamlit) → Graph-Workflow (Nodes/Edges/State) → Funktionen + LLMs → Antwort → Streamlit
```

## Warum ein Graph statt einer einfachen Pipeline?

Für simple, lineare Abläufe würde eine klassische Pipeline reichen. Sobald der Ablauf jedoch bedingte Verzweigungen, Rücksprünge oder mehrere Entscheidungspfade benötigt (komplexer Workflow), bietet ein Graph mit LangGraph die notwendige Flexibilität: Nodes können je nach State unterschiedliche Folge-Nodes ansteuern, Zustände werden explizit verwaltet, und der Ablauf bleibt nachvollziehbar und erweiterbar.

## Projektstruktur

```
15-ChatBot/
├── src/
│   └── langgraph_agentic_ai/
│       ├── graph/          # Aufbau des LangGraph-Graphen (Nodes + Edges)
│       │   └── __init__.py
│       ├── LLMs/           # LLM-Provider/-Konfiguration
│       │   └── __init__.py
│       ├── nodes/          # Node-Definitionen
│       │   └── __init__.py
│       ├── state/          # Definition des State-Objekts
│       │   └── __init__.py
│       ├── tools/          # Tools, die von Nodes/LLMs genutzt werden
│       │   └── __init__.py
│       ├── ui/             # UI-Anbindung (Streamlit)
│       ├── __init__.py
│       └── main.py         # Einstiegspunkt / Orchestrierung
├── app.py                  # Streamlit-Frontend (Chat-UI)
└── README.md
```

## Tech-Stack

- **LangChain** – LLM-Integration und Tooling
- **LangGraph** – Graph-basierte Workflow-Steuerung (Nodes, Edges, State)
- **Streamlit** – Chat-Frontend
- **LLMs** – austauschbar, je nach eingebundenem Provider

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nächste Schritte

- [ ] State-Schema definieren
- [ ] Nodes implementieren und einzeln testen
- [ ] Bedingte Edges (Routing-Logik) definieren
- [ ] Streamlit-UI an den Graphen anbinden
- [ ] End-to-End-Test des gesamten Workflows