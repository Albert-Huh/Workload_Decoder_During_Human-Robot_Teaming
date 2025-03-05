# Workload Decoder During Human-Robot Teaming

This repository provides tools and methods for decoding and analyzing human workload in collaborative human-robot environments. By monitoring human performance and physiological indicators, this project aims to enable robots to adapt their behavior in real time based on the operator’s workload, ultimately improving efficiency and safety in high-stakes or complex tasks.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Workload Inference Pipeline:** A set of scripts and utilities to process physiological data (e.g., heart rate, EEG, etc.) and behavior logs to estimate human workload.
- **Real-Time Adaptation:** Demonstration code showing how robots can adjust task allocation or behaviors based on decoded workload signals.
- **Visualization Tools:** Basic visualizations and dashboards for monitoring workload metrics over time.

## Installation
1. **Clone the repository:**
`bash
git clone https://github.com/Albert-Huh/Workload-Decoder-During_Human-Robot-Teaming.git
cd Workload-Decoder-During_Human-Robot-Teaming
`
2. **Install dependencies** (assuming a Python environment):
```bash
pip install -r requirements.txt
```
   If you don’t see a \`requirements.txt\` file, you can manually install the necessary packages as outlined in the code or documentation.

3. **(Optional) Create a virtual environment** for isolating the project’s dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Usage
1. **Data Preparation:** Organize your physiological and performance data in a folder (e.g., \`data/\`). Update any file paths in the scripts to point to your dataset.
2. **Run Analysis:** Execute the main script or notebook to preprocess and analyze your data:
```bash
python main.py
```
   or open and run any Jupyter notebooks provided.
3. **Real-Time Demo (Optional):** If the repository includes a real-time workload adaptation demo, follow the instructions in the demo folder to run it with your robot simulation or hardware.

## Data
- Ensure you follow any applicable ethical and regulatory requirements when working with human-subject data.
- To protect participant privacy, anonymize or remove identifying information before sharing data.

## Contributing
Contributions are welcome! If you have ideas for improving the workload decoding pipeline, better visualizations, or additional dataset integrations:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

## License
This project is licensed under the [MIT License](LICENSE). Please see the \`LICENSE\` file for details.
EOF
