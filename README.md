Mini Metro AI Agent
Computer vision pipeline and AI agent for autonomously playing Mini Metro — real-time state extraction via OpenCV, graph-based game representation, and hierarchical decision-making to chase world records.
How It Works

Screen capture — grabs live frames of the game
CV pipeline — detects and classifies stations, passengers, and lines using contour analysis
Graph state — represents the game as a live graph (stations as nodes, lines as edges)
Agent — hierarchical decision-making: reactive controller for emergencies, MCTS planner for weekly strategy
