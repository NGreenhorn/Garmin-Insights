# Garmin-Insights
Garmin Insights is a lightweight local app that turns years of Garmin CSV data into clear trends and personalised analysis. Built with Python, Streamlit and Pandas. Uses a local LLM via Ollama for natural-language questions. Fast, offline and ideal for making sense of your long-term 
Here’s a clean, ready to drop README in Markdown that fits your tech stack and keeps things simple and sharp.

⸻

Garmin Insights

Garmin Insights is a lightweight local app that turns years of Garmin CSV data into clear trends, patterns and personalised analysis. It runs fully offline. It lets you explore long term performance in a way Garmin often cannot. The app uses a local LLM through Ollama so you can ask natural language questions about your training history.

Features
	•	Simple CSV upload flow
	•	Automatic data cleaning and normalisation
	•	Interactive charts for pace, HR, distance, elevation and trends
	•	Natural language analysis powered by your local LLM
	•	Fast and private. Everything stays on your machine
	•	Built for runners, cyclists and anyone with rich Garmin history

Tech Stack
	•	Python
	•	Streamlit
	•	Pandas
	•	Ollama
	•	Codex style prompt layer for structured queries

Installation
	1.	Clone the repo

git clone https://github.com/ngreenhorn/garmin-insights.git
cd garmin-insights


	2.	Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate


	3.	Install dependencies

pip install -r requirements.txt


	4.	Make sure Ollama is installed and running

ollama run your-model-name



Usage

Run the Streamlit app

streamlit run app.py

Upload your exported Garmin CSV files. Explore your data. Ask questions like:
	•	“How has my average pace changed over eight years?”
	•	“Which months have the highest training load?”
	•	“Show my trends for distance and heart rate over time.”

Data Privacy

All processing happens locally. No data is sent anywhere. Your Garmin history stays on your device.

Roadmap
	•	Multi file merge improvements
	•	More advanced LLM powered insights
	•	Personalised training suggestions
	•	Fitness score aggregation

Contributing

Pull requests are welcome. Open an issue if you want to propose a feature or report a bug.

License

MIT License.

