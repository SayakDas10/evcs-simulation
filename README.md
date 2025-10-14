# evcs-simulation 

## Prerequisites

- Python 3.11 or higher.
- Git 
## Installation & Setup
1. **Clone the Repository**
```bash
git clone https://github.com/SayakDas10/evcs-simulation.git
cd evcs-simulation
``````
2. **Install uv**

This project is managed using `uv`. To install `uv`,
**macOs and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh 
```
**Windown**

```PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
After installation, you may need to restart terminal for `uv` command to be available.

3. **Install Dependencies**

````bash
uv pip sync
````
## Running the Project

```bash
uv run main.py
```
# Note:

The only thing we need to change now is the cost function. Which can be done by modifying `custom_weights.py` file. Nothing else needs to be changed.
